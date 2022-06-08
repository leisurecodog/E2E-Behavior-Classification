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

# load lane detection model & laneline detection model
yolact_model = load_yolact_edge()
yolact_model.detect.use_fast_nms = True
# PInet setting
lane_agent = Agent()
lane_agent.cuda()
lane_agent.evaluate_mode()
lane_agent.load_weights(895, "tensor(0.5546)")

OTS = functions.overtaking_system()

def inference(objdet=None, frame=None):
    global f_shape, f_type, imgsz
    draw_laneline_flag = False
    transform = FastBaseTransform()
    moving_statistics = {"conf_hist": []}
    f_shape = frame.shape
    f_type = frame.dtype
    # get lane line
    x_coord, y_coord = PInet_test(lane_agent, frame)

    center_x = f_shape[1] / 2 # get image center
    OTS.set_lane([x_coord, y_coord], center_x)
    
    # draw lane after process
    if OTS.both_lane_flag and draw_laneline_flag:
        for i in range(len(OTS.left_lane[0])):
            x, y = OTS.left_lane[:, i]
            cv2.circle(frame, (int(x), int(y)), 2, (255,255,255), 2)
        for i in range(len(OTS.right_lane[0])):
            x, y = OTS.right_lane[:, i]
            cv2.circle(frame, (int(x), int(y)), 2, (255,255,255), 2)
    # for i in range(len(objdet)):
    #     x1, y1 = objdet[i][:2]
    #     x2, y2 = objdet[i][2:4]
    #     cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255))


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

    # frame = cv2.addWeighted(lane_mask, 1, frame, 1, 0.0)
    # frame = cv2.putText(frame, OTS.msg, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
    # cv2.imshow("te", frame)
    # if cv2.waitKey(1) == 27:
        # return True
    

def set_opt():
    opt = ArgumentParser()
    opt.add_argument('--yolact_edge', action='store_false')
    opt.add_argument('--obj_det', action='store_false')
    opt.add_argument('--save_video', action='store_false')
    opt.add_argument('--video_path', type=str, default='201126152425.MOV')
    return opt.parse_args()
    
