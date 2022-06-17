
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ipdb import set_trace as debug

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=256, hidden2=256, init_w=3e-3):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fcf = nn.Linear(32, nb_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        # self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())
        # self.fcf.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.fcf(out)
        # out = self.tanh(out)
        return out

class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=512, hidden2=256, init_w=3e-3):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1+nb_actions, hidden2)
        self.fc3 = nn.Linear(hidden2, 64)
        self.fcf = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        # self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())
        # self.fcf.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, xs):
        x, a = xs
        out = self.fc1(x)
        out = self.relu(out)
        # debug()
        out = self.fc2(torch.cat([out,a],1))
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fcf(out)
        return out

class Actor_LSTM(nn.Module): # arch  from paper 2014 seq2seq Encoder
    # LSTM + Fully Connected Actor
    def __init__(self,
                input_size = 2,
                embedding_size = 256,
                hidden_size = 512,
                n_layers = 4,
                dropout = 0.1):
        super().__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.linear = nn.Linear(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, n_layers,
                           dropout = dropout)
        self.dropout = nn.Dropout(dropout)
        
        self.Flatten = nn.Flatten()
        fn_size = hidden_size * n_layers
        self.linear2 = nn.Linear(fn_size, fn_size//2)
        self.linear3 = nn.Linear(fn_size//2, fn_size//4)
        self.linear4 = nn.Linear(fn_size//4, fn_size//8)
        self.linear5 = nn.Linear(fn_size//8, fn_size//16)
        self.linear6 = nn.Linear(fn_size//16, 2)
    def forward(self, x):
        """
        x: input batch data, size: [sequence len, batch size, feature size]
        for the argoverse trajectory data, size(x) is [20, batch size, 2]
        """
        # embedded: [sequence len, batch size, embedding size]
        # print(x.shape) # x: [batch size, input size * len traj]
        x = torch.reshape(x, (x.shape[0], -1, 2)) 
        # x: [batch size, len traj, input size]
        x = torch.permute(x,(1, 0, 2))
        # x: [len traj, batch size, input size]
        embedded = self.dropout(F.relu(self.linear(x)))
        # you can checkout https://pytorch.org/docs/stable/nn.html?highlight=lstm#torch.nn.LSTM
        # for details of the return tensor
        # briefly speaking, output coontains the output of last layer for each time step
        # hidden and cell contains the last time step hidden and cell state of each layer
        # we only use hidden and cell as context to feed into decoder
        output, (hidden, cell) = self.rnn(embedded)
        # hidden = [n layers * n directions, batch size, hidden size]
        # cell = [n layers * n directions, batch size, hidden size]
        # the n direction is 1 since we are not using bidirectional RNNs
        # print("h, c=============", output.shape, hidden.shape, cell.shape)
        hidden = torch.permute(hidden,(1, 0, 2))
        out = self.Flatten(hidden)
        # print(out.shape)
        out = self.linear2(out)
        out = self.linear3(out)
        out = self.linear4(out)
        out = self.linear5(out)
        out = self.linear6(out)
        # print(out.shape)
        # return hidden, cell
        return out