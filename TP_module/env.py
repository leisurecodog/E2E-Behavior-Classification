from email import utils
from operator import truediv
from re import S
from urllib.request import proxy_bypass
import numpy as np
import cv2
import random
import math
from TP_module.utils.data import *
from sklearn.model_selection import train_test_split

class environment:
    def __init__(self, data_select='train', nb_t=5, mode='default', args=None):
        self.nb_t = nb_t
        self.nb_s = self.nb_t * 2
        self.nb_a = 2
        self.data_select = data_select
        self.teacher_forcing_ratio = 1.01
        self.args = args
        self.current_s = [0, 0]
        self.prob = 1.0
        self.idx = {}
        self.idx['train'] = -1
        self.idx['valid'] = -1
        self.idx['test'] = -1
        self.mode = 'train'
        self.discount_factor = 0.9
        self.decay_long = 20000
        self.tf_decay = 1.0 / self.args.train_iter
        self.dataset = {}
        if mode == 'default':
            dataset = 'bdd100k'
            self.datas = self.anonymous(data_preprocess(dataset), nb_t)
            if self.args.preprocess == 'interpolation':
                print("=============Make Interpolation=============")
                self.datas = self.interpolation(self.datas)
            self.split_dataset()
            # self.des_trajs = self.dataset[data_select]
            # self.des_idx = 0
            # self.traj_idx = 0
            self.traj_idx_arr = {}
            self.shuffle()
    def sigmoid(self, x, k=0.1):
        return 1 / (1+ np.exp(-x/k))

    def split_dataset(self):
        self.dataset['train'], self.dataset['test'] = train_test_split(self.datas, train_size=0.8)
        self.dataset['valid'], self.dataset['test'] = train_test_split(self.dataset['test'], train_size=0.5)

    def shuffle(self):
        self.traj_idx_arr['train'] = np.arange(len(self.dataset['train']))
        np.random.shuffle(self.traj_idx_arr['train'])
        self.traj_idx_arr['valid'] = np.arange(len(self.dataset['valid']))
        np.random.shuffle(self.traj_idx_arr['valid'])
        self.traj_idx_arr['test'] = np.arange(len(self.dataset['test']))
        np.random.shuffle(self.traj_idx_arr['test'])

    def RMSE(self, pred, gt):
        return np.sqrt(np.sum(np.square(np.array(gt) - np.array(pred))))
    
    def MSE(self, pred, gt):
        return np.sum(np.square(np.array(gt) - np.array(pred)))

    def MAE(self, pred, gt):
        loss_abs = np.abs(gt - pred)
        return np.sum(loss_abs)

    def teacher_forcing(self, n_s, gt_n_s, decay=False):
        if decay:
            self.teacher_forcing_ratio -= self.tf_decay
            # print("Decay teacher forcing ratio:", self.teacher_focing_ratio)
        self.prob = self.sigmoid(self.teacher_forcing_ratio)
        self.prob = max(self.prob, 0.25)
        teacher_force = random.random() < self.prob
        
        if self.mode != 'train': # Only using teacher forcing when training stage
            teacher_force = False

        return  np.concatenate((n_s[:-2], np.asarray(gt_n_s, dtype=np.float32)), axis=0) if teacher_force else n_s

    def get_next_state(self, act):
        # example
        # [x1, y1, x2, y2, x3, y3 .... ,xn, yn]
        # x, y = xn, yn
        x, y = self.current_s[-2:]
        # xn+1 , yn+1
        x += act[0]
        y += act[1]
        # res = [x1, y1, x2, y2, x3, y3 .... ,xn+1, yn+1]
        res = np.concatenate((self.current_s[2:], np.asarray([x, y], dtype=np.float32)), axis=0)
        return res

    def reset(self):
        # if mode == 'test':
        #     print("Starting Test Mode.....")
        self.des_trajs = self.dataset[self.mode]
        self.traj_idx = 0
        self.idx[self.mode] += 1
        self.done = False

        if self.idx[self.mode] >= len(self.traj_idx_arr[self.mode]):
            np.random.shuffle(self.traj_idx_arr[self.mode])
            self.idx[self.mode] = 0

        self.des_idx = self.traj_idx_arr[self.mode][self.idx[self.mode]] # which trajectory
        # print(self.des_idx, mode)
        self.current_s = np.array(self.des_trajs[self.des_idx][:self.nb_t]).flatten()
        self.traj_idx = self.nb_t - 1 # time t in trajectory
        return self.current_s
        # self.current_s = self.des_trajs[self.des_idx][self.traj_idx]
        
    
    def step(self, act, step=None):
        n_s = self.get_next_state(act) # the next_state actor did.
        self.traj_idx += 1
        # get ground truth next_state
        gt_n_state = self.des_trajs[self.des_idx][self.traj_idx]
        # calculate error (reward)
        reward = -self.RMSE(n_s[-2:], gt_n_state)
        # reward = -self.MSE(n_s[-2:], gt_n_state)
        # reward = -self.MAE(n_s[-2:], gt_n_state)

        # do teacher forcing selection
        decay_flag = True if self.mode == 'train' else False
        
        n_s = self.teacher_forcing(n_s, gt_n_state, decay=True)
        self.current_s = n_s
        if self.traj_idx >= len(self.des_trajs[self.des_idx])-1:
            self.done = True

        return n_s, reward, self.done, None

    def anonymous(self, dics, t):
        total_traj = []
        for i in range(len(dics)):
            for k, v in dics[i].items():
                if len(v) < t*2:
                    continue
                total_traj.append(v)
        return total_traj
    def interpolation(self, data):
        inter_trajs = []
        for i in range(len(data)):
            inter_traj = []
            
            for j in range(len(data[i])-1):
                midx = (data[i][j][0] + data[i][j+1][0]) / 2
                midy = (data[i][j][1] + data[i][j+1][1]) / 2
                inter_traj.append(data[i][j])
                inter_traj.append([midx, midy])
                if j == len(data[i])-2:
                    inter_traj.append(data[i][j+1])
            # for j in range(10):
            #     print(data[i][j], inter_traj[j])
            #     input()

            inter_trajs.append(inter_traj)
        return inter_trajs
    # ===================================================================================