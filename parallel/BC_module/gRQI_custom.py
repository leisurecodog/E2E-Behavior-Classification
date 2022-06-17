from cProfile import label
from cgitb import reset
import enum
from genericpath import exists
from http.client import UNSUPPORTED_MEDIA_TYPE
from imp import new_module
from locale import normalize
from math import sqrt , pow
import heapq
from operator import itemgetter
from termios import TIOCSERCONFIG
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

from BC_module.gRQI_main import * 
from BC_module.ML import *
from BC_module.data_preprocess import *


def RQI(Laplacian_Matrices):
    U_Matrices = []
    from scipy import linalg as LA
    for Lis_for_each_video in Laplacian_Matrices:
        time_start_all = time.time()
        # ListofUs = []
        for L_index,L in enumerate(Lis_for_each_video):
            if first_laplacian(L_index):
                Lambda_prev, U_prev = la.eig(L) # need top k eigenvectors
                Lambda_prev = np.diag(np.real(Lambda_prev))
                # Lambda_prev = Lambda_prev[0:10,0:10]
                U_prev= np.real(U_prev)
            else:
                U_curr, Lambda = GraphRQI(U_prev, Lis_for_each_video, L_index, Lambda_prev)
                Lambda_prev = Lambda
                # ListofUs.append(U_curr)
                U_prev = U_curr[-1]
            # Daeig , Va , X = LA.svd ( L , lapack_driver='gesvd' )
        # print("time for computing spectrum for one video: ", (time.time() - time_start_all))
        U_Matrices.append(U_curr[0])
    return U_Matrices

def visualization(X, y, method='TSNE'):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import manifold, datasets
    #Prepare the data
    
    # digits = datasets.load_digits(n_class=6)
    # X, y = digits.data, digits.target

    # n_samples, n_features = X.shape
    # n = 20  
    # img = np.zeros((10 * n, 10 * n))
    # for i in range(n):
    #     ix = 10 * i + 1
    #     for j in range(n):
    #         iy = 10 * j + 1
    #         img[ix:ix + 8, iy:iy + 8] = X[i * n + j].reshape((8, 8))
    # plt.figure(figsize=(8, 8))
    # plt.imshow(img, cmap=plt.cm.binary)
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()

    #t-SNE
    if method == 'TSNE':
        X_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=1).fit_transform(X)

        #Data Visualization
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne - x_min) / (x_max - x_min)  #Normalize
        plt.figure(figsize=(8, 8))
        for i in range(X_norm.shape[0]):
            plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]), 
                    fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([])
        plt.yticks([])
        plt.show()
    elif method == 'PCA':
        from sklearn.decomposition import PCA
        pca=PCA(n_components=2)
        pca.fit(X).transform(X)

def main(opt):
    
    data_name = 'total_data'
    label_name = 'total_label'
    aggressive_name = 'aggressive_id'
    appear_id_name = 'appear_id'
    if opt.ext_mode == 1:
        data_name = 'total_data'
        label_name = 'total_label'
    elif opt.ext_mode == 2:
        data_name = 'total_data_full'
        label_name = 'total_labe_full'
    # videos_list, labels_list = create_label_and_data(mot_path, label_path, opt)
    videos_list = []
    labels_list = []
    aggressive_id = []
    appear_id = []
    if opt.dataset == 'own':
        # old path
        if opt.half1:
            label_path = '/home/rvl/Desktop/fengan/Dataset/bdd100k/bdd100k/bdd100k/behavior_gt_new'
            mot_path = '/home/rvl/Desktop/fengan/Dataset/bdd100k/bdd100k/bdd100k/labels_with_ids_offset/track/train/'
        elif opt.half2:
            label_path = '/home/fengan/Desktop/Dataset/BDD100K MOT/bdd100k_ano_half/behavior_gt'
            mot_path = '/home/fengan/Desktop/Dataset/BDD100K MOT/bdd100k_ano_half/labels_with_ids_offset/track/train/'
        else:
            label_path = '/home/fengan/Desktop/Dataset/BDD100K MOT/bdd100k_full_ab/behavior_gt'
            mot_path = '/home/fengan/Desktop/Dataset/BDD100K MOT/bdd100k_full_ab/labels_with_ids_offset/track/train/'

        if opt.load and os.path.exists(data_name + '.npy') and os.path.exists(label_name + '.npy'):
            videos_list = list(np.load(data_name + '.npy', allow_pickle=True))
            labels_list = list(np.load(label_name + '.npy', allow_pickle=True))
        else:
            videos_list, labels_list, aggressive_id, appear_id = create_data_and_label(mot_path, label_path, opt)
            if opt.old:
                videos_list, labels_list = create_label_and_data(mot_path, label_path, opt)
                data_name = 'total_data_old'
                label_name = 'total_label_old'
    elif opt.dataset == 'meteor':
        mot_path = '/home/fengan/Desktop/Dataset/METEOR/MOT labels_with_ids'
        label_path = '/home/fengan/Desktop/Dataset/METEOR/driving behavior labels original'
        videos_list, labels_list, aggressive_id, appear_id = create_data_and_label(mot_path, label_path, opt)
    elif opt.dataset == 'argo':
        videos_list = np.load('data/argo/argo_data.npy',allow_pickle=True)
        labels_list = np.load ( 'data/argo/argo_labels.npy', allow_pickle=True )
        # print(len(labels_list[0]))
        for i in range(len(labels_list[0])):
            if labels_list[0][i][1] == 0.0 or labels_list[0][i][1] == 1.0 or labels_list[0][i][1] == 2.0:
                labels_list[0][i][1] = 1.0
            else:
                labels_list[0][i][1] = 0.0
    np.save(aggressive_name, np.array(aggressive_id))
    np.save(appear_id_name, np.array(appear_id))

    if opt.vid_num >= 100 and opt.dataset != 'argo': # save data
        np.save(data_name, np.array(videos_list))
        np.save(label_name, np.array(labels_list))
    # print(np.shape(videos_list), np.shape(labels_list))    
    Adjacency_Matrices = computeA(videos_list, labels_list, opt.neighber, opt.dataset , True)
    Laplacian_Matrices = extractLi(Adjacency_Matrices)
    # =================================================================================================


    U_name = 'full_u_matrices'
    U_Matrices = []
    if not os.path.exists(U_name + '.npy') and opt.ext_mode == 2 and opt.vid_num >= 100:
        U_Matrices = RQI(Laplacian_Matrices)
        np.save(U_name, np.array(U_Matrices))
    elif opt.ext_mode == 2:
        U_Matrices = list(np.load(U_name + '.npy', allow_pickle=True))
    else:
        U_Matrices = RQI(Laplacian_Matrices)
    # U_Matrices = np.array(U_Matrices)
    # print(U_Matrices)
    print("np.shape(U_Matrices): ", np.shape(U_Matrices))
    if opt.dataset == 'argo':
        new_U_Matrices = U_Matrices[0]
    else:
        if opt.ext_mode == 1:
            new_U_Matrices = np.reshape(U_Matrices, (-1, opt.id_num))
        elif opt.ext_mode == 4:
            new_U_Matrices = np.reshape(U_Matrices, (-1, opt.id_num))
        elif opt.ext_mode == 2:
            aggressive_id = list(np.load(aggressive_name + '.npy', allow_pickle=True))
            appear_id = list(np.load(appear_id_name + '.npy', allow_pickle=True))
            new_U_Matrices, labels_list = post_extract_id(U_Matrices, aggressive_id, appear_id, labels_list, opt)
            new_U_Matrices = np.reshape(new_U_Matrices, (-1, opt.s1_thre))

    new_embedding = new_U_Matrices
    # print(labels_list)
    labels = []
    if opt.dataset == 'argo':
        for j in range(len(labels_list[0])):
            labels.append(labels_list[0][j][1])
    else:
        if opt.ext_mode == 1 or opt.ext_mode == 2 or opt.ext_mode == 4:
            for ll in labels_list:
                for v in ll[0].values():
                    labels.append(v)

    if opt.visualize:
        visualization(new_embedding, labels)
    if opt.original:
        data = {}
        from sklearn.neural_network import MLPClassifier as MLP
        mlp = MLP( hidden_layer_sizes=(10,50), max_iter=4000)
        Xtrain, Xtest, ytrain, ytest = train_test_split(new_embedding, labels, test_size=0.1)
        data['c_prec'] = 0
        data['c_recall'] = 0
        data['a_prec'] = 0
        data['a_recall'] = 0
        data['acc'] = 0
        mlp.fit(Xtrain, ytrain)
        y_pred = mlp.predict(Xtest)
        from ML import show_metrics
        show_metrics(ytest, y_pred, True)

    if opt.dl:
        pytorch_train(new_embedding, labels, opt)

    if opt.anomaly:
        four_data = [{},{}]
        for idx in range(2):
            four_data[idx]['c_prec'] = 0
            four_data[idx]['c_recall'] = 0
            four_data[idx]['a_prec'] = 0
            four_data[idx]['a_recall'] = 0
            four_data[idx]['acc'] = 0
        times = 1
        for i in range(times):
            Xtrain, Xtest, ytrain, ytest = train_test_split(new_embedding, labels, test_size=0.1)
            save_list = []
            if opt.oversampling:
                Xtrain, ytrain = Oversampling(Xtrain, ytrain)

            d1 = osvm_train_test(Xtrain, ytrain, Xtest, ytest)
            d2 = isolation_forest_train_test(Xtrain, ytrain, Xtest, ytest)
            SUOD_train_test(Xtrain, ytrain, Xtest, ytest)

            save_list.append(d1)
            save_list.append(d2)
            for idx, d in enumerate(save_list):
                four_data[idx]['c_prec'] += d['conservative']['precision']
                four_data[idx]['c_recall'] +=  d['conservative']['recall']
                four_data[idx]['a_prec'] += d['aggressive']['precision']
                four_data[idx]['a_recall'] += d['aggressive']['recall']
                if 'accuracy' in d.keys():
                    four_data[idx]['acc'] += d['accuracy']
        for idx in range(len(save_list)):
            print("{:.3f} {:.3f} {:.3f} {:.3f} {:.3f}".format(four_data[idx]['c_prec']/times, four_data[idx]['c_recall']/times, four_data[idx]['a_prec']/times, four_data[idx]['a_recall']/times, four_data[idx]['acc']/times))
    if opt.ml:
        four_data = [{},{},{},{}]
        for idx in range(4):
            four_data[idx]['c_prec'] = 0
            four_data[idx]['c_recall'] = 0
            four_data[idx]['a_prec'] = 0
            four_data[idx]['a_recall'] = 0
            four_data[idx]['acc'] = 0
        times = 10
        for i in range(times):
            Xtrain, Xtest, ytrain, ytest = train_test_split(new_embedding, labels, test_size=0.1)
            # print(np.shape(ytest))
            save_list = []
            if opt.oversampling:
                Xtrain, ytrain = Oversampling(Xtrain, ytrain)
                # print(np.shape(ytrain))
            d1 = xgboost_train_test(Xtrain, ytrain, Xtest, ytest)
            # d2, d3 = imb_xgboost_train_test(Xtrain, ytrain, Xtest, ytest)
            d4 = rf_train_test(Xtrain, ytrain, Xtest, ytest)
            save_list.append(d1)
            # save_list.append(d2)
            # save_list.append(d3)
            save_list.append(d4)
            for idx, d in enumerate(save_list):
                # print(d)
                four_data[idx]['c_prec'] += d['conservative']['precision']
                four_data[idx]['c_recall'] +=  d['conservative']['recall']
                four_data[idx]['a_prec'] += d['aggressive']['precision']
                four_data[idx]['a_recall'] += d['aggressive']['recall']
                if 'accuracy' in d.keys():
                    four_data[idx]['acc'] += d['accuracy']
        for idx in range(4):
            print("{:.3f} {:.3f} {:.3f} {:.3f} {:.3f}".format(four_data[idx]['c_prec']/times, four_data[idx]['c_recall']/times, four_data[idx]['a_prec']/times, four_data[idx]['a_recall']/times, four_data[idx]['acc']/times))

if __name__ == "__main__":

    from parser import add_parser
    opt = add_parser()
    if opt.dataset == 'meteor':
        opt.gap = 5
    main(opt)
