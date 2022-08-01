import torch.nn as nn
import torch
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        # self.hidden1 = nn.Linear(input_size, 10)
        # self.hidden2 = nn.Linear(10, 8)
        # self.output = nn.Linear(8, 2)
        self.passUpDown =  nn.Sequential(
             nn.Conv1d(1, 512, 3, 1, 1),
             nn.ReLU(),
             nn.Conv1d(512, 1, 3, 1, 1),
             nn.ReLU()
        )
        self.l1 = nn.Linear(10, 2)

    
    def forward(self, x):
        # x = F.relu(self.hidden1(x))
        # x = F.relu(self.hidden2(x))
        # x = F.relu(self.hidden3(x))
        # x = F.relu(self.hidden4(x))
        # x = F.relu(self.hidden5(x))
        # x = F.relu(self.hidden6(x))
        # x = F.relu(self.hidden7(x))
        # x = F.softmax(self.output(x))
        updown = self.passUpDown(x)
        x = torch.add(x, updown)
        updown = self.passUpDown(x)
        x = torch.add(x, updown)
        updown = self.passUpDown(x)
        x = torch.add(x, updown)
        updown = self.passUpDown(x)
        x = torch.add(x, updown)
        updown = self.passUpDown(x)
        x = torch.add(x, updown)
        x = torch.flatten(x, 1)
        x = F.softmax(self.l1(x))
        return x


