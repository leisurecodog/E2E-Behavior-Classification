#############################################################################################################
##
##  Source code for testing
##
#############################################################################################################

import cv2
import json
import torch
from OT_module.PInet import agent
import numpy as np
from copy import deepcopy
import time
from OT_module.PInet.parameters import Parameters
import sys,os
from OT_module.PInet.ransac import Line_ransac
from os import path


p = Parameters()

###############################################################
##
## Training
## 
###############################################################
def Testing():
    print('Testing')

    ##############################
    ## Get agent and model
    ##############################
    print('Get agent')
    if p.model_path == "":
        lane_agent = agent.Agent()
    else:
        lane_agent = agent.Agent()
        lane_agent.load_weights(895, "tensor(0.5546)")
    ##############################
    ## Check GPU
    ##############################
    print('Setup GPU mode')
    if torch.cuda.is_available():
        lane_agent.cuda()

    ##############################
    ## testing
    ##############################
    print('Testing loop')
    lane_agent.evaluate_mode()

    
def PInet_test(lane_agent, test_images, thresh = p.threshold_point):

    resize_images = cv2.resize(test_images, (512, 256))
    resize_images = resize_images/255.0
    resize_images = np.rollaxis(resize_images, axis=2, start=0)
    resize_images=np.array([resize_images])

    result = lane_agent.predict_lanes_test(resize_images)
    confidences, offsets, instances = result[-1]
    
    num_batch = len(resize_images)

    out_x = []
    out_y = []
    out_images = []

    for i in range(1):


        ratio_h = test_images.shape[0] / resize_images.shape[2]
        ratio_w = test_images.shape[1] / resize_images.shape[3]


        # test on test data set
        image = deepcopy(resize_images[i])
        image =  np.rollaxis(image, axis=2, start=0)
        image =  np.rollaxis(image, axis=2, start=0)*255.0
        image = image.astype(np.uint8).copy()
        #print(type(p.grid_y),type(p.grid_x))
        confidence = confidences[i].view(int(p.grid_y), int(p.grid_x)).cpu().data.numpy()

        offset = offsets[i].cpu().data.numpy()
        offset = np.rollaxis(offset, axis=2, start=0)
        offset = np.rollaxis(offset, axis=2, start=0)
        
        instance = instances[i].cpu().data.numpy()
        instance = np.rollaxis(instance, axis=2, start=0)
        instance = np.rollaxis(instance, axis=2, start=0)

        # generate point and cluster
        raw_x, raw_y = generate_result(confidence, offset, instance, thresh)

        # eliminate fewer points
        in_x, in_y = eliminate_fewer_points(raw_x, raw_y)
                
        # sort points along y 
        in_x, in_y = sort_along_y(in_x, in_y)  
        #in_x, in_y = eliminate_out(in_x, in_y, confidence, deepcopy(image))
        #in_x, in_y = util.sort_along_y(in_x, in_y)
        #in_x, in_y = eliminate_fewer_points(in_x, in_y)
        #print(in_x,in_y)
        in_x, in_y = conversion_coordinate(in_x, in_y, ratio_w, ratio_h)
        # in_x, in_y=Line_ransac(in_x, in_y, test_images)
        try:
            in_x, in_y = Line_ransac(in_x, in_y, test_images)
        except:
            in_x = None 
            in_y = None

    return in_x, in_y


############################################################################
## post processing for eliminating outliers
############################################################################
def conversion_coordinate(x, y, ratio_w, ratio_h):
    for i in range(len(x)):
        for j in range(len(x[i])):
            x[i][j] = int(np.round(x[i][j] * ratio_w))
            y[i][j] = int(np.round(y[i][j] * ratio_h))
    return x, y

############################################################################
## eliminate result that has fewer points than threshold
############################################################################
def eliminate_fewer_points(x, y):
    # eliminate fewer points
    out_x = []
    out_y = []
    for i, j in zip(x, y):
        if len(i)>2:
            out_x.append(i)
            out_y.append(j)     
    return out_x, out_y   

############################################################################
## generate raw output
############################################################################
def generate_result(confidance, offsets,instance, thresh):

    mask = confidance > thresh
    #print(mask)

    grid = p.grid_location[mask]
    offset = offsets[mask]
    feature = instance[mask]

    lane_feature = []
    x = []
    y = []
    for i in range(len(grid)):
        if (np.sum(feature[i]**2))>=0:
            point_x = int((offset[i][0]+grid[i][0])*p.resize_ratio)
            point_y = int((offset[i][1]+grid[i][1])*p.resize_ratio)
            if point_x > p.x_size or point_x < 0 or point_y > p.y_size or point_y < 0:
                continue
            if len(lane_feature) == 0:
                lane_feature.append(feature[i])
                x.append([])
                x[0].append(point_x)
                y.append([])
                y[0].append(point_y)
            else:
                flag = 0
                index = 0
                for feature_idx, j in enumerate(lane_feature):
                    index += 1
                    if index >= 12:
                        index = 12
                    if np.linalg.norm((feature[i] - j)**2) <= p.threshold_instance:
                        lane_feature[feature_idx] = (j*len(x[index-1]) + feature[i])/(len(x[index-1])+1)
                        x[index-1].append(point_x)
                        y[index-1].append(point_y)
                        flag = 1
                        break
                if flag == 0:
                    lane_feature.append(feature[i])
                    x.append([])
                    x[index].append(point_x) 
                    y.append([])
                    y[index].append(point_y)
                
    return x, y

def draw_points(x, y, image):
    color_index = 0
    for i, j in zip(x, y):
        color_index += 1
        if color_index > 12:
            color_index = 12
        for index in range(len(i)):
            image = cv2.circle(image, (int(i[index]), int(j[index])), 1, p.color[color_index], 2)

    return image

def sort_along_y(x, y):
    out_x = []
    out_y = []

    for i, j in zip(x, y):
        i = np.array(i)
        j = np.array(j)

        ind = np.argsort(j, axis=0)
        out_x.append(np.take_along_axis(i, ind[::-1], axis=0).tolist())
        out_y.append(np.take_along_axis(j, ind[::-1], axis=0).tolist())
    
    return out_x, out_y

# if __name__ == '__main__':
#     Testing()
