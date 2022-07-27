# -*- coding: UTF-8 -*-
import cv2
import os
from os import listdir, getcwd
from os.path import join
import numpy as np
import glob
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import warnings

warnings.filterwarnings('ignore')
   
class PolynomialRegression(object):
    def __init__(self, degree=2, coeffs=None):
        self.degree = degree
        self.coeffs = coeffs

    def fit(self, X, y):
        self.coeffs = np.polyfit(X.ravel(), y, self.degree)

    def get_params(self, deep=False):
        return {'coeffs': self.coeffs}

    def set_params(self, coeffs=None, random_state=None):
        self.coeffs = coeffs

    def predict(self, X):
        poly_eqn = np.poly1d(self.coeffs)
        y_hat = poly_eqn(X.ravel())
        return y_hat
    
    def score(self, X, y):
        return mean_squared_error(y, self.predict(X))

def is_Horizontal_line(y,h,mid):
    for i in range(mid,h,10):
        line_count=0
        for j in range(0,10):
            line_count=line_count+len(np.where(y==i+j)[0])
            if len(np.where(y==i+j)[0])>4:
                return 1
        #print(line_count)
        if line_count>25:
            return 1

def Line_ransac(x_point, y_point, img):
    result_x = []
    result_y = []    
    for lane_x,lane_y in zip(x_point, y_point):
        x=np.array(deepcopy(lane_x))
        y=np.array(deepcopy(lane_y))
        if len(x)<5:
            continue
        ransac = RANSACRegressor(PolynomialRegression(degree=2),
                 residual_threshold=1 * np.std(y),
                 random_state=1)
        ransac.fit(np.expand_dims(y, axis=1), x)
        # inlier_mask = ransac.inlier_mask_
        yfit=np.linspace(int(img.shape[0]/2),img.shape[0],int(img.shape[1]/2))
        xfit= ransac.predict(np.expand_dims(yfit, axis=1))
        eq_coefficient=ransac.estimator_.coeffs
        #if is_Horizontal_line(y,img.shape[0],int(img.shape[0]/2)):
        #    continue
######判斷遞增或遞減
        Flag_diff=0
        issorted= sum(np.sign(np.diff(xfit)))
        if issorted!=len(xfit)-1 and issorted!=-(len(xfit)-1):
#                    if max(xfit)-min(xfit)>60:                        
            if np.sign(eq_coefficient[0])<0 and max(xfit)-min(xfit)>60 :                    
                max_x=np.array(np.where(xfit==np.max(xfit)))
                candidate_yfit_1=np.copy(yfit[0:min(max_x[0,:])])
                candidate_xfit_1=np.copy(xfit[0:min(max_x[0,:])])
                candidate_yfit_2=np.copy(yfit[min(max_x[0,:]):])
                candidate_xfit_2=np.copy(xfit[min(max_x[0,:]):])
                Flag_diff=1
            elif np.sign(eq_coefficient[0])>0 and min(xfit)-max(xfit)<-60:
                min_x=np.array(np.where(xfit==np.min(xfit)))
                candidate_yfit_1=np.copy(yfit[0:min(min_x[0,:])])
                candidate_xfit_1=np.copy(xfit[0:min(min_x[0,:])])
                candidate_yfit_2=np.copy(yfit[min(min_x[0,:]):])
                candidate_xfit_2=np.copy(xfit[min(min_x[0,:]):])      
                Flag_diff=1
            if Flag_diff==1:
                Q1x=candidate_xfit_1[0]-candidate_xfit_1[len(candidate_xfit_1)-1]
                Q1y=candidate_yfit_1[len(candidate_yfit_1)-1]-candidate_yfit_1[0]
                Q1c=candidate_xfit_1[len(candidate_xfit_1)-1]*candidate_yfit_1[0]-candidate_xfit_1[0]*candidate_yfit_1[len(candidate_yfit_1)-1]
                
                Q2x=candidate_xfit_2[0]-candidate_xfit_2[len(candidate_xfit_2)-1]
                Q2y=candidate_yfit_2[len(candidate_yfit_2)-1]-candidate_yfit_2[0]
                Q2c=candidate_xfit_2[len(candidate_xfit_2)-1]*candidate_yfit_2[0]-candidate_xfit_2[0]*candidate_yfit_2[len(candidate_yfit_2)-1]
                P=np.array((np.hstack((x[:,np.newaxis],y[:,np.newaxis]))))
                dis1=np.fabs(Q1y*P[:,0]+Q1x*P[:,1]+Q1c)/(np.power(Q1x*Q1x+Q1y*Q1y,0.5))
                dis2=np.fabs(Q2y*P[:,0]+Q2x*P[:,1]+Q2c)/(np.power(Q2x*Q2x+Q2y*Q2y,0.5))
                
                if sum(dis1)<sum(dis2):
                    xfit=candidate_xfit_1
                    yfit=candidate_yfit_1
                else:
                    xfit=candidate_xfit_2
                    yfit=candidate_yfit_2
        # xy=np.array((np.hstack((xfit[:,np.newaxis],yfit[:,np.newaxis]))),dtype=int)
        # total_xy.append(xy)
        result_x.append(xfit)
        result_y.append(yfit) 

    return result_x,result_y
