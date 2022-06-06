from os import ftruncate
import numpy as np
import OT_module.functions as functions
import cv2
import torch
import math
import os, sys
import time

from argparse import ArgumentParser
# import yolact edge module
from OT_module.yolact_edge_project.utils.timer import start
from OT_module.yolact_edge_project.eval import load_yolact_edge, prep_display
from OT_module.yolact_edge_project.utils.augmentations import FastBaseTransform
# ======================================================================================
# import PInet module
from OT_module.PInet.test import PInet_test
from OT_module.PInet.agent import Agent

gradientThreshold = 0.5
my_arrow = functions.Arrow()
# my_lane = functions.Lane()
OTS = functions.overtaking_system()
f_type = []
f_shape = []
bb_list = []
opt = []
def customArrow(wid, hei):
    global f_type, f_shape, my_arrow, my_lane
    arrow_mat = np.zeros(f_shape, f_type)
    # arrowArr = np.array((1,3))
    tmp = my_arrow.get_shape()
    pts = []
    pts.append(tmp)
    cv2.fillPoly(arrow_mat, np.array([pts],dtype=np.int32), my_arrow.get_color(my_lane.missing_lane()))
    arrow_center = my_arrow.get_center()
    r = cv2.getRotationMatrix2D(arrow_center, my_arrow.get_rotate_angle(), 1)
    res_mat = cv2.warpAffine(arrow_mat, r, (arrow_mat.shape[1], arrow_mat.shape[0]))
    return res_mat

def mask_for_front_lane(img): # create make for lane detection
    height, width = img.shape
    rightX = functions.get_another_xPoint(0.5, 0, height, height/2 * 3)
    leftX = functions.get_another_xPoint(-0.5, width, height, height/2 * 3)

    #polygon mask for lane
    pts = np.array([
        [width/5*4, height],
        [width/5, height], 
        [leftX - leftX/10, height/2 + height/10],
        [rightX, height/2 + height/10]])
    return pts

def apply_mask(img, pts):
    filter_color = 255
    mask = np.zeros((img.shape[0], img.shape[1]),img.dtype)
    # fillPoly needs the points arrays are int32, so need to convert its dtype.
    cv2.fillPoly(mask, np.array([pts],dtype=np.int32), filter_color)
    return mask

def HoughTransformP(canny_img):
    global my_lane
    lines_new = cv2.HoughLinesP(canny_img, 1, np.pi/180.0, 100, 50, 10)
    # cv::HoughLinesP(in img, double rho, double theta, int thres, minLineLen, maxLineGap)
    hei = canny_img.shape[0]
    # int width = CannyMat.cols;
    ratio = 0.6
    leftPts = []
    rightPts = []
    if lines_new is not None:
        for i in range(len(lines_new)):
            L = lines_new[i][0]
            gradientResult = (L[1]-L[3])/(L[0]-L[2])
            if math.isinf(gradientResult):
                gradientResult = 100000
            # filtering line according to threshold of line gradient.
            if(abs(gradientResult) < gradientThreshold):
                continue
            
            firstX = functions.get_another_xPoint(gradientResult, L[0], L[1], hei)
            secX = functions.get_another_xPoint(gradientResult, L[0], L[1], int(hei*ratio))
            
            pt1 = [firstX, hei]
            pt2 = [secX, int(hei*ratio)]
            
            if(gradientResult < 0):
                leftPts.append(pt1)
                leftPts.append(pt2)
            
            elif(gradientResult > 0):
                rightPts.append(pt1)
                rightPts.append(pt2)
        
        my_lane.update_lane_point(leftPts, rightPts)

def lane_detection(img):
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gaussian_img = cv2.GaussianBlur(gray_img,(3,3),0)
    mask = mask_for_front_lane(gaussian_img)
    final_mask = apply_mask(gaussian_img, mask)
    canny_img = cv2.Canny(gaussian_img, 0, 255)
    masked_img = cv2.bitwise_and(final_mask, canny_img)
    HoughTransformP(masked_img)
    # return masked_img

def draw():
    global f_shape, f_type, my_lane, my_arrow
    hei, width, c = f_shape
    total_draw = np.zeros(f_shape, f_type)
    '''
    lane_point:
    left lane     [[x1,y1], [x2,y2],
    right lane    [x3,y3], [x4,y4]]
    '''
    lane_points = my_lane.get_lane_point() # get left and right lane line.

    # draw lines.
    for i in range(0, len(lane_points),2):
        cv2.line(total_draw, (int(lane_points[i][0]),int(lane_points[i][1])), (int(lane_points[i+1][0]), int(lane_points[i+1][1])), (255,255,0), 10)

    # draw arrow
    if(my_arrow.height == 0): # check arrow is exist or not.
        my_arrow.set_WH(wid=width/20, hei=hei/20)
    if(my_arrow is not None):
        
        my_arrow.set_position([width/2, int(hei - hei/4)])
        if(my_lane.missing_lane() is False):
        
            tmpX = (lane_points[0][0] + lane_points[2][0]) / 2.0
            tmpY = hei / 2 * 3
            angle = np.arctan2((tmpX - (width/2)), (tmpY-(hei - hei/4)))
            angle = angle * (180/3.1415926)
            my_arrow.set_rotate_angle(round(-angle)*5)
            my_arrow.update_rotate_angle()
        
        arrowMat = customArrow(width, hei); # get arrowMat.
        arrowRatio = 0.7
        res = cv2.addWeighted(total_draw, 1, arrowMat, arrowRatio, 1)
    else:
        res = total_draw  
    total_draw = cv2.addWeighted(res, 1, total_draw, 0.3, 0.0)
    return total_draw


def inference(yolact_model=None, objdet=None, frame=None):
    global f_shape, f_type, imgsz
    
    yolact_model.detect.use_fast_nms = True

    # PInet setting
    lane_agent = Agent()
    lane_agent.cuda()
    lane_agent.evaluate_mode()
    lane_agent.load_weights(895, "tensor(0.5546)")

    transform = FastBaseTransform()
    moving_statistics = {"conf_hist": []}
    f_shape = frame.shape
    f_type = frame.dtype
    # get lane line
    x_coord, y_coord = PInet_test(lane_agent, frame)
    center_x = f_shape[1] / 2 # get image center

    OTS.set_lane([x_coord, y_coord], center_x)
    
    # draw lane after process
    if OTS.both_lane_flag and True:
        for i in range(len(OTS.left_lane[0])):
            x, y = OTS.left_lane[:, i]
            cv2.circle(frame, (int(x), int(y)), 2, (255,255,255), 2)
        for i in range(len(OTS.right_lane[0])):
            x, y = OTS.right_lane[:, i]
            cv2.circle(frame, (int(x), int(y)), 2, (255,255,255), 2)
    for i in range(len(objdet)):
        x1, y1 = objdet[i][:2]
        x2, y2 = objdet[i][2:4]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255))


    start_time = time.time() # calculate performance time

    frame_tensor = torch.from_numpy(frame).cuda().float()
    batch = FastBaseTransform()(frame_tensor.unsqueeze(0))
    extras = {"backbone": "full", "interrupt": False, "keep_statistics": False,"moving_statistics": moving_statistics}
    with torch.no_grad():
        net_outs = yolact_model(batch, extras=extras) # yolact edge detect lane mask
    preds = net_outs["pred_outs"]
    # get lane mask
    lane_mask = prep_display(preds, frame_tensor, None, None, undo_transform=False, class_color=True)
    OTS.detect_overtaking(objdet, lane_mask)
    print("=====================", OTS.msg)
    res = cv2.addWeighted(lane_mask, 1, frame, 1, 0.0)
    cv2.imshow("te", res)
    if cv2.waitKey(100) == 27:
        return True
    

def set_opt():
    opt = ArgumentParser()
    opt.add_argument('--yolact_edge', action='store_false')
    opt.add_argument('--obj_det', action='store_false')
    opt.add_argument('--save_video', action='store_false')
    opt.add_argument('--video_path', type=str, default='201126152425.MOV')
    return opt.parse_args()
    
# def main():
#     global opt
#     yolo_v5 = load_yolov5() if opt.obj_det else None
#     yolact_edge = load_yolact_edge() if opt.yolact_edge else None
#     prefix = '/media/rvl/D/Work/fengan/Dataset/CEO/front'
#     inference(yolact_model=yolact_edge, objdet=yolo_v5, video_path=os.path.join(prefix, opt.video_path))

# if __name__ == '__main__':
    # opt = set_opt()
    # # print(opt)
    # main()
