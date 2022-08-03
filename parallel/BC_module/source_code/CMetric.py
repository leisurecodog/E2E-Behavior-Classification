import enum
from http.client import UNSUPPORTED_MEDIA_TYPE
from termios import TIOCSERCONFIG
import numpy as np

from numpy import *
import scipy.io
# from sklearn.datasets import make_classification
from matplotlib.pyplot import *
# from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier as MLP
from sklearn import svm
import seaborn as sns
sns.set()
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix

from numpy import linalg as la

import time
# from sklearn import preprocessing
from gRQI_main import * 
from ML import *
from data_preprocess import *
import networkx as nx
MU = 500
def compute_Adj(videos_list):
    Adj_vids = []
    G_vids = []
    for video in videos_list:
        Adj_vid = []
        G_vid = []
        for frame in video:
            G_f = nx.Graph()
            Adj = np.zeros((len(frame), len(frame)))
            values = list(frame.values())
            # print(values[0])
            # input()
            len_v = len(values)
            for i in range(len_v):
                for j in range(len_v):
                    if i == j:
                        continue
                    xi, yi = values[i]
                    xj, yj = values[j]
                    dist = computeDist(xi, yi, xj, yj)
                    if dist < MU:
                        Adj[i,j] = dist   
                        G_f.add_edge(i, j, weight=dist)
            Adj_vid.append(Adj)
            G_vid.append(G_f)
        Adj_vids.append(Adj_vid)
        G_vids.append(G_vid)
    return Adj_vids, G_vids

def speed_related(ts_prev, ts_curr):
    res = []
    if ts_prev[1] > ts_curr[1]:
        res = 1
    elif ts_prev[1] < ts_curr[1]:
        res = -1
    else:
        res = 0
    return res

def cal_speed(videos_list):
    speed_vids = []
    for video in videos_list:
        speed_vid = []
        for idx, frame in enumerate(video):
            if idx == 0:
                continue
            prev_id_list = list(video[idx-1].keys())
            curr_id_list = list(frame.keys())
            speed_dict = {}
            for ptv in prev_id_list:
                for ptc in curr_id_list:
                    if ptv == ptc:
                        rt_speed = speed_related(video[idx-1][ptv], frame[ptc])
                        speed_dict[ptc] = rt_speed
            speed_vid.append(speed_dict)
        speed_vids.append(speed_vid)
    return speed_vids

def main(opt):
    videos_list = []
    labels_list = []
    if opt.dataset == 'own':
        # old path
        # if opt.half1:
        label_path = '/media/rvl/D/Work/fengan/Dataset/bdd100k/bdd100k/bdd100k/behavior_gt_new'
        mot_path = '/media/rvl/D/Work/fengan/Dataset/bdd100k/bdd100k/bdd100k/labels_with_ids_offset/track/train/'
        videos_list, labels_list, _, _ = create_data_and_label(mot_path, label_path, opt)
    speeds_list = cal_speed(videos_list)

    Adjacency_Matrices, G_vids = compute_Adj(videos_list)
    # Laplacian_Matrices = extractLi(Adjacency_Matrices)
    print(np.shape(Adjacency_Matrices))
    # ================================CMetric Implementation Start==============================
    CCs = list()
    DCs = list()
    # vids_shortest = []
    for vid_idx, G_vid in enumerate(G_vids):
        # vid_shortest = []
        CC = list()
        DC = list()
        for time_t, G_f in enumerate(G_vid):
            # Calculate & save shortest path using dictionary.
            
            shortest_path_list_f = []
            # print(len(Adjacency_Matrices[vid_idx][idx]))
            cc = dict()
            dc = dict()
            N = len(Adjacency_Matrices[vid_idx][time_t])
            for i in range(N):
                shortest_path_list_f = []
                for j in range(i+1, N):
                    if i == j:
                        continue
                    shortest_path_list_f.append(nx.shortest_path_length(G_f, source=i, target=j, weight='weight')) 
                s_sum = np.sum(shortest_path_list_f)
                cc[i] = (N-1) / s_sum if s_sum != 0 else 0
            CC.append(cc[i])
            for i in range(N):
                dc[i] = []
                print(speeds_list[vid_idx][time_t])
                print(Adjacency_Matrices[vid_idx][time_t+1][i])
                Nor = [idx for idx, edge in enumerate(Adjacency_Matrices[vid_idx][time_t][i,:])\
                     if edge != 0 and speeds_list[vid_idx][time_t][idx] <= 0]
                # print(Nor)
                if time_t == 0:
                    dc[i].append(len(Nor))
                else:
                    dc[i].append(len(Nor) + dc[i][-1])
            DC.append(dc)
        DCs.append(DC)
        CCs.append(CC)
    

if __name__ == "__main__":
    from parser import add_parser
    opt = add_parser()
    # dics = {}
    # for id_setting in range(5, 100, 5):
        # opt.id_num = id_setting
        # (x, y, z) = main(opt)
        # dics[id_setting] = [x, y, z]
    # print("ADJ time, Laplacian time, RQI time:")
    # for k, v in dics.items():
        # print("{}: {}".format(k, v))
    if opt.dataset == 'meteor':
        opt.gap = 5
    main(opt)
