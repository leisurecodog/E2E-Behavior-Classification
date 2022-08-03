from cProfile import label
from logging.config import valid_ident
from math import sqrt , pow
import heapq
from operator import itemgetter
import numpy as np
from numpy import *
import scipy.io
from sklearn.datasets import make_classification
from matplotlib import pyplot as plt
from matplotlib.pyplot import *
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier as MLP
from sklearn import svm
import seaborn as sns
sns.set()
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
from numpy import linalg as la
from math import pow
import time
from sklearn import preprocessing
import os
from collections import Counter
# ===========================GraphRQI=======================================

import argparse
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
import cv2
# ===========================ByteTrack======================================

from numpy import random
from models.experimental import attempt_load
from utils.general import check_img_size, plot_one_box,set_logging
from utils.torch_utils import select_device
import yolo_detect
# ===========================Yolov5=========================================

def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument(
        "demo", default="video", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        #"--path", default="./datasets/mot/train/MOT17-05-FRCNN/img1", help="path to images or video"
        "--path", default="./videos/1.MOV", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )
    # exp file
    # parser.add_argument(
    #     "-f",
    #     "--exp_file",
    #     default=None,
    #     type=str,
    #     help="pls input your expriment description file",
    # )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.4, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    parser.add_argument('--show', action="store_true")
    return parser

def load_yolov5():
    global names, colors, imgsz
    weights = 'weights/s_best.pt'
    set_logging()
    device = select_device('')

    # load yolov5 model
    model_v5 = attempt_load(weights, map_location=device)
    # get image size for feed of yolov5
    imgsz = check_img_size(640, s=model_v5.stride.max())
    
    # check cpu utility
    half = device.type != 'cpu'
    if half:
        model_v5.half()

    names = model_v5.module.names if hasattr(model_v5, 'module') else model_v5.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    return model_v5

def SM( u, v,Ainv):
    time_1SM = time.time()
    term1 =  Ainv@u
    term3 = v@Ainv
    term2 = np.eye(u.shape[1]) + term3@u
    term2inv = np.array([[term3[1,1],-1*term3[0,1]],[-1*term3[1,0],term3[0,0]]])
    # print ( "time for computing one SM op: " , time.time () - time_1SM )
    return Ainv -(term1@term2inv)@term3

def DU( sigma, U_prev, mu, Lam_prev):
    time_1DU = time.time()
    # d = [Lam-mu for Lam in Lam_prev]
    # d = [d[i]-sigma[i] for i in range(len(d))]
    # D = np.diag(d)

    D = Lam_prev + np.abs(sigma) - mu*np.eye(len(sigma))
    Ut = np.transpose(U_prev)
    for i in range(Ut.shape[0]):
        Ut[i][i] = Ut[i,i]/pow(la.norm(U_prev[:,i]),2)

    # print ( "time for computing one DU op: " , time.time () - time_1DU )
    # return (U_prev@D[0:])@(Ut)
    return (U_prev@D)@(Ut)

def form_block(A, mu_j):
    time_1bm = time.time()
    A = (1-mu_j)*A
    A = np.hstack(  ( A, np.zeros([A.shape[0],1]) )  )
    block_matrix = np.vstack(  (  A,np.zeros([1,A.shape[1]]) )  )
    # print ( "time for computing one bm op: " , time.time () - time_1bm )
    block_matrix[-1][-1] = 1-mu_j
    return block_matrix

def first_laplacian(index):
    return True if index==0 else False

def extractLi(A):
    listofTRAFLis = []
    listofLis = []
    T = A.shape[0]
    for i in range(T-2):
        a = A[0:i+3,0:i+3]
        d = [np.sum(a[l,:]) for l in range(a.shape[0])]  # da = sum(A,2);
        D = np.diag ( d )  # Da = diag(da);
        L = D - a
        listofLis.append(L)
    return listofLis

def computeDist ( x1 , y1 , x2 , y2 ):
    return sqrt ( pow ( x1 - x2 , 2 ) + pow ( y1 - y2 , 2 ) )

def computeKNN ( curr_dict , ID , k, dataset ):

    if dataset == 'traf':
        ID_x , ID_y = curr_dict[ ID ]
        dists = {}
        for j in list ( curr_dict.keys () ):
            if j != ID:
                dists[ j ] = computeDist ( curr_dict[ ID ][ 0 ] , curr_dict[ ID ][ 1 ] , curr_dict[ j ][ 0 ] , curr_dict[ j ][ 1 ] )
        KNN_IDs = dict ( heapq.nsmallest ( k , dists.items () , key=itemgetter ( 1 ) ) )
        neighbors = list ( KNN_IDs.keys () )
    # print(ID,'==',list(KNN_IDs.keys()))
    else:
        lis = [el[0] for el in curr_dict]
        ind = lis.index(ID)
        ID_x , ID_y = curr_dict[ ind ][1]
        dists = {}
        for j in lis:
            if j != ID:
                dists[ j ] = computeDist ( curr_dict[ ind ][1][ 0 ] , curr_dict[ ind][1][ 1 ] , curr_dict[ lis.index(j) ][1][ 0 ] ,curr_dict[ lis.index(j) ][1][ 1 ] )
        KNN_IDs = dict ( heapq.nsmallest ( k , dists.items () , key=itemgetter ( 1 ) ) )
        neighbors = list ( KNN_IDs.keys () )

    return neighbors

def computeA(frame, label_list, nbrs, dataset):
    A_size = max(label_list)
    A = np.zeros([int(A_size), int(A_size)])
    for idx, id in enumerate(label_list):
        if id < A_size:
            if id in list(frame.keys()):
                neighbors = computeKNN(frame, id, nbrs, dataset)
            for neighbor in neighbors:
                if neighbor in label_list:
                    if idx < label_list.index(neighbor):
                        A[idx][label_list.index(neighbor)] = 1
    return A
            
def GraphRQI ( U_prev, now_Laplacian, prev_Laplacian, Lambda_prev):
    U = []
    # L = Lis_for_each_video[L_index]
    # L_prev = Lis_for_each_video[ L_index - 1 ]
    L = now_Laplacian
    L_prev = prev_Laplacian
    Lambda_curr = np.zeros([L.shape[0],L.shape[0]])
    print("U_prev.shape = ", U_prev.shape)
    print("L.shape = ", L.shape)
    delta = L[0:-1,-1]
    print("delta.shape = ", delta.shape)
    delta = np.expand_dims(delta, axis=1)
    print("expanded delta.shape = ", delta.shape)
    # Compute sigma, sigmaTranspose
    
    sigma = np.eye(delta.shape[0])
    for i in range(delta.shape[0]):
        sigma[i,i] = delta[i,0]
    print("sigma.shape = ", sigma.shape)
    delta = np.hstack((delta,np.zeros([delta.shape[0],1])))
    delta = np.vstack((delta, np.zeros([1,delta.shape[1]])))
    delta[-1,-1]=1
    print("stacked delta.shape = ", delta.shape)

    deltaTranspose = delta.T
    times = []
    for j in range (L.shape[0]):
        if j<4:
            times.append(time.time())
        mu_j = Lambda_prev[j,j] if j != L.shape[0]-1 else Lambda_prev[j-1,j-1]
        x_old = np.random.rand(L.shape[0],1)
        x_new = x_old / la.norm ( x_old )
        print(sigma.shape, U_prev.shape)
        x_new = SM ( delta , deltaTranspose , form_block ( DU ( sigma , U_prev , mu_j , Lambda_prev ) , mu_j )) @ x_old

        # while not converged(x_old, x_new):
        for i in range(1):
        # Perform the rqi iterations to compute u_j
            x_old = x_new/la.norm(x_new)
        # x_new = UPDATE*x_old
            x_new = SM(delta, deltaTranspose, form_block(DU(sigma, U_prev, mu_j, Lambda_prev),mu_j))@x_old

        u_j = x_new/la.norm(x_new)
        Lambda_curr[j,j] = (u_j.T@(L@u_j)).item()
        U.append(u_j)

    U = np.array(U).T
    # Lambda_curr[-1,-1] = 1

    return U, Lambda_curr

def behavior_inference(res):
    # modify the graphRQI to frame by frame handle.
    # 1. load data or get data from MOT model
    video_list = []
    video_list.append(res)
    default_dataset = 'traf'
    nbrs = 4
    label_list = []
    dict_item = {} # res: [frame_id, target id, x, y, w, h, score, -1, -1, -1]
    for l in res:
        dict_item[int(l[1])] = [int((l[2]+l[4])/2), int((l[3]+l[5])/2)]
        label_list.append(int(l[1]))

    Adjacency_Matrices = computeA(dict_item, label_list, nbrs, default_dataset)
    Laplacian_Matrices = extractLi(Adjacency_Matrices)
    for L in Laplacian_Matrices:
        print("loop L.shape = ", L.shape)
        input()
    # print(np.array(Laplacian_Matrices[0]).shape)
    #====================================GraphRQI Algo=============================
    U_Matrices = []
    from scipy import linalg as LA
    print("frame_id = ", res[0][0]-1)
    print("Laplacian_Matrices shape = ", Laplacian_Matrices[0].shape)
    if first_laplacian(res[0][0]-1):
        behavior_inference.Lambda_prev, behavior_inference.U_prev = la.eig(Laplacian_Matrices[0])
        print("U_prev_size.shape = ", behavior_inference.U_prev.shape)

        behavior_inference.Lambda_prev = np.diag(np.real(behavior_inference.Lambda_prev))
        behavior_inference.U_prev = np.real(behavior_inference.U_prev)
        print("U_prev_real_size.shape = ", behavior_inference.U_prev.shape)

    else:
        U_curr, Lambda = GraphRQI(behavior_inference.U_prev, Laplacian_Matrices[0], behavior_inference.prev_Laplacian_Matrices, behavior_inference.Lambda_prev)
        behavior_inference.Lambda_prev = Lambda
        # ListofUs.append(U_curr)
        behavior_inference.U_prev = U_curr[-1]
    behavior_inference.prev_Laplacian_Matrices = Laplacian_Matrices
    # 3. graphRQI
    
    # 4. plot the result
def main():
    global names, colors, imgsz
    args = make_parser().parse_args()
    cap = cv2.VideoCapture(args.path)
    tracker = BYTETracker(args, frame_rate=30)
    detector = load_yolov5()
    frame_id = 1
    # results = []
    while True:
        val, frame = cap.read()
        results = []
        if val:
            img_info = {}
            outputs = yolo_detect.detect(detector, imgsz, names, frame)
            img_info['height'], img_info['width'] = frame.shape[:2]
            img_info['raw_img'] = frame.copy()
            if outputs is not None:
                track_result = tracker.update(reversed(outputs), [img_info['height'], img_info['width']], [img_info['height'], 
                img_info['width']])
               
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in track_result:
                    tlwh = t.tlwh
                    tid = t.track_id
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    # results.append(
                    #     f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1"
                    # )
                    results.append([frame_id, tid, tlwh[0], tlwh[1], tlwh[2], tlwh[3], t.score, -1, -1, -1])
                if args.show:
                    online_im = plot_tracking(
                        img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=0)
                    cv2.imshow('ttt', online_im)
                    if cv2.waitKey(10) == 27: # end play when press ESC button.
                        return
                behavior_inference(results)
            else:
                online_im = img_info['raw_img']
            
        else:
            break
        frame_id += 1



if __name__ == '__main__':
    main()