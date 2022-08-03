import numpy as np
import argparse
from copy import deepcopy
import torch
from utils.data import *

# from normalized_env import NormalizedEnv
from evaluator import Evaluator
from ddpg import DDPG
from util import *
import env as Env
import parser
import cv2

def Displacement_Error(pred, GT, s_len):
    ade = 0
    fde = 0
    for j in range(len(pred)):
        err = np.sqrt(np.sum(np.square(np.array(pred[j]) - np.array(GT[j+s_len]))))
        ade += err
        fde = err
    ade = ade / len(pred)
    # print(res)
    return ade, fde

def get_next_state(s, a):
    return [s[0] + a[0], s[1] + a[1]]

def start_demo(trajs, policy, s_len):
    # print("s_len", s_len)
    total_ADE = 0
    total_FDE = 0
    traj_num = len(trajs)
    for i in range(traj_num):
        # print(trajs[i])
        traj_pred = []
        traj_len = len(trajs[i])
        next_traj = []
        if traj_len < s_len:
            continue
        for j in range(traj_len-s_len):
            np_traj = np.array(trajs[i])
            if j == 0:
                traj_state = np_traj[j:j+s_len].flatten()
            else:
                # print(type(traj_state[2:]), np.array(next_traj))
                traj_state = np.concatenate((traj_state[2:], np.array(next_traj)))
            action = policy(traj_state)
            next_traj = get_next_state(traj_state[-2:], action)
            traj_pred.append(next_traj)
            
        ADE, FDE = Displacement_Error(traj_pred, trajs[i], s_len)
        total_ADE += ADE
        total_FDE += FDE
        # for j in range(len(traj_pred)):
        #     print("timestamp J:", j)
        #     print("pred: ", traj_pred[j])
        #     print("GT: ",  trajs[i][j+s_len])
        #     input()
    print(traj_num)
    print("Total_ADE: {:.2f}, Total_FDE: {:.2f}".format(total_ADE / traj_num, total_FDE / traj_num))

def demo(folder_name):
    # dataset = 'kitti'
    # trajs = data_preprocess(dataset)
    # print(trajs[0])
    traj_infer_path = '/media/rvl/D/Work/fengan/Dataset/INFER-datasets/kitti/targetGT'
    dirs = sorted(os.listdir(traj_infer_path))
    trajs = []
    for dir in dirs:
        p = os.path.join(traj_infer_path, dir)
        dirr = sorted(os.listdir(p))
        traj = []
        for dd in dirr:
            fp = os.path.join(p, dd)
            img_p = os.listdir(fp)
            if len(img_p) == 0:
                continue
            fp = os.path.join(fp, img_p[0])
            img = cv2.imread(fp)
            gt = np.unravel_index(np.argmax(img), img.shape)
            # print(gt[:2])
            traj.append([gt[0], gt[1]])
        trajs.append(traj)
    
    args = parser.get_parser()
    # init env & model
    env = Env.environment(mode='demo', args=args)
    nb_states = env.nb_s
    nb_actions = env.nb_a
    nb_t = env.nb_t
    agent = DDPG(nb_states, nb_actions, args)

    # model setting
    model_path = folder_name
    agent.load_weights(model_path)
    agent.is_training = False
    agent.eval()
    # get policy
    policy = lambda x: agent.select_action(x, decay_epsilon=False)
    # total_traj = env.anonymous(trajs, nb_t)
    total_traj = trajs
    start_demo(total_traj, policy, nb_t)

def inference(trajs, policy, s_len):
    trajs_pred = []
    
    for i in range(len(trajs)):
        traj_pred = []
        next_traj = []
        for j in range(len(trajs[i])):
            if j == s_len: # predict how long for each trajectory
                break
            np_traj = np.array(trajs[i]) # i-th trajectory
            if j == 0:
                traj_state = np_traj[j:j+s_len].flatten()
            else:
                traj_state = np.concatenate((traj_state[2:], np.array(next_traj)))
            action = policy(traj_state)
            next_traj = get_next_state(traj_state[-2:], action)
            traj_pred.append(next_traj)
        trajs_pred.append(traj_pred)
    return trajs_pred

if __name__ == '__main__':
    demo('seq2seq')