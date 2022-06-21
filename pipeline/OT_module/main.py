from os import ftruncate
import numpy as np
import OT_module.functions as functions
import cv2
import torch
import math
import os, sys
import time
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

import torch.multiprocessing as mp
manager = mp.Manager()
return_dict = manager.dict()

def inference(objdet, frame):
    st = time.time()
    # draw_laneline_flag = True
    moving_statistics = {"conf_hist": []}
    # frame.shape: [Height, Width, Channels]
    center_x = frame.shape[1] / 2 # get image center
    # get lane line

    x_coord, y_coord = PInet_test(lane_agent, frame)
    
    if x_coord is not None and y_coord is not None and len(x_coord) > 0 and len(y_coord) > 0:
        OTS.set_lane([x_coord, y_coord], center_x)
    
    # draw lane after process
    # if OTS.both_lane_flag and draw_laneline_flag:
        # for i in range(len(OTS.left_lane[0])):
            # x, y = OTS.left_lane[:, i]
            # cv2.circle(frame, (int(x), int(y)), 2, (255,255,255), 2)
        # for i in range(len(OTS.right_lane[0])):
            # x, y = OTS.right_lane[:, i]
            # cv2.circle(frame, (int(x), int(y)), 2, (255,255,255), 2)
    # for i in range(len(objdet)):
        # x1, y1 = objdet[i][:2]
        # x2, y2 = objdet[i][2:4]
        # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255))

    start_time = time.time() # calculate performance time
    
    frame_tensor = torch.from_numpy(frame).cuda().float()
    batch = FastBaseTransform()(frame_tensor.unsqueeze(0))
    extras = {"backbone": "full", "interrupt": False, "keep_statistics": False,"moving_statistics": moving_statistics}

    with torch.no_grad():
        net_outs = yolact_model(batch, extras=extras) # yolact edge detect lane mask
    preds = net_outs["pred_outs"]
    # get lane mask
    lane_mask = prep_display(preds, frame_tensor, None, None, undo_transform=False, class_color=True)
    
    if OTS.both_lane_flag:
        # t1 = time.time()
        OTS.detect_overtaking(objdet, lane_mask, frame)
        # t2 = time.time()
        # print("Overtaking time: ", t2-t1)
    else:
        OTS.msg = "You can't overtake."

    return OTS.msg
    # frame = cv2.addWeighted(lane_mask, 1, frame, 1, 0.0)
    # cv2.putText(frame, OTS.msg, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    
    # if OTS.both_lane_flag:
    #     print("OT time: ", time.time()-st)
    # return frame
    
