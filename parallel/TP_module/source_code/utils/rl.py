import numpy as np
from tqdm import tqdm, trange
from multiprocessing import Pool
from torch.autograd import Variable
import torch
import math
from numba import jit
from utils.visualize import visualize_svf

def find_expert_SVF_1d(traj, vtp, opt):
    traj_num, traj_len = traj.shape[:2]
    svf = np.zeros((vtp.state_num))
    for i in range(traj_num):
        # print(traj[i])
        for j in range(traj_len):
            svf[vtp.xy2idx(traj[i][j])] += 1
    return svf

def find_expected_svf_1d(policy, traj, vtp, opt):
    print(traj.shape)
    traj_num, traj_len = traj.shape[:2]
    # traj_len = traj.shape[1]
    state_num = vtp.state_num
    start_state_count = np.zeros(state_num)
    # count svf for start state
    for i in range(traj_num):
        start_state_count[vtp.xy2idx(traj[i][0])] += 1
        

    svf = np.tile(start_state_count, (traj_len, 1)).T
    for i in range(traj_num):
        for t in range(1, traj_len):
            svf[:, t] = 0
            for s in range(state_num):
                # print(s)
                for idx in range(len(vtp.actions)):
                    n_state = vtp.next_state_idx(s, idx)
                    svf[n_state, t] += (svf[s, t-1] * policy[s, idx])
    return svf.sum(axis=1)

def find_policy(value, reward, vtp, opt):

    reward_1d = reward.flatten()

    total_state = vtp.state_num
    total_action = len(vtp.actions)
    Q = np.zeros((total_state, total_action))
    for s in range(total_state):
        for a in range(total_action):
            next_s = vtp.next_state_idx(s, a)
            # if vtp.over_boundry_idx(next_s):
            #     next_s = vtp.clip(next_s)
            Q[s, a] = reward_1d[next_s] + vtp.discount_factor * value[next_s]
    Q -= Q.max(axis=1).reshape((total_state, 1))  # For numerical stability
    Q = np.exp(Q*20) / np.exp(Q*20).sum(axis=1).reshape((total_state, 1))  # softmax over actions
    return Q
# @jit
def find_value(reward, vtp, opt, threshold=1e-0):
    # value iteration start
    # reward_1d = reward.flatten().cpu().detach().numpy()

    reward_1d = reward.flatten()
    V = np.zeros(vtp.state_num)
    delta = np.inf
    while delta > threshold:
        delta = 0.0
        for state in range(vtp.state_num):
            next_s_list = [vtp.next_state_idx(state, a) for a in range(len(vtp.actions))]

            new_v = reward_1d[state] + max([vtp.discount_factor * V[ss] for ss in next_s_list])
            delta = max(delta, abs(V[state] - new_v))
            V[state] = new_v
        # print(delta)
    return V
    
def compute_nll(policy, demo_traj, vtp):
    # nll: Negative Log Likelihood
    # print("NLL calculate is started.")
    # print(policy)
    
    import warnings
    nlls = []
    for num in range(demo_traj.shape[0]):
        prob = 1.0
        for i in range(demo_traj.shape[1] - 1):
            action = vtp.get_action(demo_traj[num][i], demo_traj[num][i + 1])
            if action is None:
                raise RuntimeError('no action can move from {} to {}'.format(demo_traj[num][i], demo_traj[num][i + 1]))

            # state = vtp.xy_to_idx((demo_traj[i, 0], demo_traj[i, 1]))
            # state = int(demo_traj[num][i, 0]*vtp.grid_size + demo_traj[num][i, 1])
            state = vtp.xy2idx(demo_traj[num][i])
            if isinstance(policy, np.ndarray):
                prob *= policy[state, action]
            else:
                prob *= policy.choose_action(state, vtp, prob=True)[action]
        try:
            nll = -math.log(prob) / demo_traj.shape[1]
            nlls.append(nll)
        except ValueError:
            print("ValueError Happen, show Prob: {}".format(prob))
    return np.mean(nlls)

def rl(idx, reward, future_traj, vtp, opt):
    # print("RL process is running.")
    
    svf_demo = find_expert_SVF_1d(future_traj, vtp, opt)
    value = find_value(reward, vtp, opt)
    policy = find_policy(value, reward, vtp, opt)
    print("IDX:", idx)
    expect_svf = find_expected_svf_1d(policy, future_traj, vtp, opt)
    
    # nll_sample = compute_nll(policy, future_traj, vtp)
    svf_diff = svf_demo - expect_svf
    visualize_svf(np.reshape(svf_demo, (vtp.grid_size, -1)), "expert_svf_{}.jpg".format(idx), opt)
    visualize_svf(np.reshape(expect_svf, (vtp.grid_size, -1)), "expect_svf_{}.jpg".format(idx), opt)
    visualize_svf(np.reshape(svf_diff, (vtp.grid_size, -1)), "svf_diff_{}.jpg".format(idx), opt)
    visualize_svf(np.reshape(value, (vtp.grid_size, -1)), "value_{}.jpg".format(idx), opt)
    visualize_svf(reward[0], "reward_{}.jpg".format(idx), opt)
    svf_diff = np.reshape(svf_diff, (vtp.grid_size, -1))
    svf_diff = Variable(torch.from_numpy(svf_diff).float(), requires_grad=False)
    return svf_diff, False

def rl_DQN(reward, future_traj, policy, vtp, opt):
    svf_demo = find_expert_SVF_1d(future_traj, vtp, opt)
    expect_svf = find_expected_svf_1d(policy, future_traj, vtp, opt) #bottleNeck
    # nll_sample = compute_nll(DQN, future_traj, vtp)
    
    svf_diff = svf_demo - expect_svf
    svf_diff = Variable(torch.from_numpy(svf_diff).float(), requires_grad=False)

    # print("DONE")
    # return svf_diff, nll_sample
    return svf_diff, False

def parallel_process(rewards, future_trajs, DQN, vtp, opt):
    n_sample = rewards.size()[0]
    # print(n_sample)
    result = []
    pool = Pool(processes=n_sample)
    # policy = np.array([DQN.choose_action(torch.Tensor(np.array(vtp.idx2xy(i))), vtp, prob=True) for i in range(vtp.state_num)])
    
    for i in range(n_sample):
        
        reward = rewards[i]
        reward = reward.cpu().detach().numpy()
        future_traj = future_trajs[i]
        future_traj = future_traj.cpu().detach().numpy()
        if opt.feature_dataset == 'outside':
            future_traj = np.array([future_traj[~np.isnan(future_traj).any(axis=2)]])

        result.append(pool.apply_async(rl, args=(i, reward, future_traj, vtp, opt)))
        # result.append(pool.apply_async(rl_DQN, args=(reward, future_traj, policy, vtp, opt)))
        
    pool.close()
    pool.join()

    res = [[result[i].get()[0]] for i in range(n_sample)]
    svf_diff_var = torch.cat([torch.unsqueeze(torch.Tensor(res[i][0]), 0) for i in range(n_sample)])
    # svf_demos = [[result[i].get()[1]] for i in range(n_sample)]
    # expect_svfs = [[result[i].get()[2]] for i in range(n_sample)]
    # print(svf_diff_var.shape)
    # nll_list = [np.array([result[i].get()[1]]) for i in range(n_sample)]
    
    return svf_diff_var