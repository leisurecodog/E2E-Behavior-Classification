
import numpy as np
import sys, os
import torch
import torch.nn as nn
from torch.optim import Adam

from TP_module.model import (Actor, Critic, Actor_LSTM)
from TP_module.memory import SequentialMemory
from TP_module.random_process import OrnsteinUhlenbeckProcess
from TP_module.util import *

# from ipdb import set_trace as debug
def load_Aseq():
    from TP_module.ASeq2Seq import Encoder, Attention, Decoder, ASeq2Seq
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    INPUT_DIM = 2
    OUTPUT_DIM = 2
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    ENC_HID_DIM = 512
    DEC_HID_DIM = 512
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

    model = ASeq2Seq(enc, dec, device).to(device)

    attn_t = Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc_t = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    dec_t = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

    model_t = ASeq2Seq(enc, dec, device).to(device)
    return model, model_t

def load_seq2seq():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from TP_module.seq2seq import Seq2Seq, Encoder, Decoder
    INPUT_DIM = 2
    OUTPUT_DIM = 2
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    actor = Seq2Seq(enc, dec, device)

    enc_trg = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec_trg = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    actor_target = Seq2Seq(enc_trg, dec_trg, device)
    return actor, actor_target

criterion = nn.MSELoss()

class DDPG(object):
    def __init__(self, nb_states, nb_actions, args):
        
        if args.seed > 0:
            self.seed(args.seed)

        self.nb_states = nb_states
        self.nb_actions= nb_actions
        
        # Create Actor and Critic Network
        net_cfg = {
            'hidden1':args.hidden1, 
            'hidden2':args.hidden2, 
            'init_w':args.init_w
        }
        if args.actor == 'LSTM':
            self.actor = Actor_LSTM()
            self.actor_target = Actor_LSTM()
        elif args.actor == 'seq2seq':
            self.actor, self.actor_target = load_seq2seq()
        elif args.actor == 'aseq':
            self.actor, self.actor_target = load_Aseq()
        else:
            self.actor = Actor(self.nb_states, self.nb_actions, **net_cfg)
            self.actor_target = Actor(self.nb_states, self.nb_actions, **net_cfg)

        self.actor_optim  = Adam(self.actor.parameters(), lr=args.prate)

        self.critic = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_target = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_optim  = Adam(self.critic.parameters(), lr=args.rate)

        hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)
        
        #Create replay buffer
        self.memory = SequentialMemory(limit=args.rmsize, window_length=args.window_length)
        self.random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=args.ou_theta, mu=args.ou_mu, sigma=args.ou_sigma)

        # Hyper-parameters
        self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon

        # 
        self.epsilon = 1.0
        self.s_t = None # Most recent state
        self.a_t = None # Most recent action
        self.is_training = True

        # 
        if USE_CUDA: self.cuda()

    def update_policy(self):
        # Sample batch
        state_batch, action_batch, reward_batch, \
        next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)

        # Prepare for the target q batch
        next_q_values = self.critic_target([
            to_tensor(next_state_batch, volatile=True),
            self.actor_target(to_tensor(next_state_batch, volatile=True)),
        ])
        # next_q_values.volatile=False

        target_q_batch = to_tensor(reward_batch) + \
            self.discount*to_tensor(terminal_batch.astype(np.float))*next_q_values

        # Critic update
        self.critic.zero_grad()
        
        # print(state_batch.dtype, state_batch, action_batch.dtype, action_batch)
        q_batch = self.critic([ to_tensor(state_batch), to_tensor(action_batch) ])
        
        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()

        policy_loss = -self.critic([
            to_tensor(state_batch),
            self.actor(to_tensor(state_batch))
        ])

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def observe(self, r_t, s_t1, done):
        if self.is_training:
            self.memory.append(self.s_t, self.a_t, r_t, done)
            self.s_t = s_t1

    def random_action(self):
        action = np.random.uniform(-1.,1.,self.nb_actions)
        self.a_t = action
        return action

    def select_action(self, s_t, decay_epsilon=True):
        action = to_numpy(
            self.actor(to_tensor(np.array([s_t])))
        ).squeeze(0)
        action += self.is_training*max(self.epsilon, 0)*self.random_process.sample()
        # action = np.clip(action, -1., 1)

        if decay_epsilon:
            self.epsilon -= self.depsilon
        
        self.a_t = action
        return action

    def reset(self, obs):
        self.s_t = obs
        self.random_process.reset_states()

    def load_weights(self, output):
        if output is None: return
        select_str = ''
        self.actor.load_state_dict(
            torch.load('{}/actor{}.pkl'.format(output, select_str))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic{}.pkl'.format(output, select_str))
        )


    def save_model(self, output, best=False):
        if best:
            torch.save(
                self.actor.state_dict(),
                '{}/actor_best.pkl'.format(output)
            )
            torch.save(
                self.critic.state_dict(),
                '{}/critic_best.pkl'.format(output)
            )
        else:
            torch.save(
                self.actor.state_dict(),
                '{}/actor.pkl'.format(output)
            )
            torch.save(
                self.critic.state_dict(),
                '{}/critic.pkl'.format(output)
            )

    def seed(self, s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)
