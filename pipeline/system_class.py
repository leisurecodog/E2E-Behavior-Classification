import os
import logging
import numpy as np
import cv2
from TP_module.util import prGreen
import torch
import time
from MOT_module.MOT import MOT
from TP_module.TP import TP
from BC_module.BC import BC
from OT_module.OT import OT

class DrivingBehaviorSystem:
    def __init__(self, args):
        """
        self.traj: list = [frame_1, frame_2, ... , frame_i]
        frame_i: dict = {key: id, value: [x,y] coordinate}
        
        MOT: Multiple Object Tracking
        TP: Trajectory Prediction
        BC: Behaivior Classification
        OT: OverTaking Assistant
        """
        self.reset = False
        self.MOT = MOT()
        self.TP = TP()
        self.BC = BC(self.TP.traj_len_required)
        self.OT = OT()
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.f_id = 0
        self.writer = cv2.VideoWriter('/media/rvl/D/Work/fengan/code/system/output.avi', fourcc, 20.0, (1080, 720))
        self.avg_fps = 0
    # ========================= Other small function code ==========================
    def show(self, frame, t_total=None):
        bbox = self.MOT.result
        MOT_show_flag = False
        future_traj_flag = True if self.TP.future_trajs is not None else False
        BC_result_flag = True if self.BC.result is not None else False

        # Draw bounding box & agent ID and Behavior Classification Result
        # draw_color_idx: 0 mean normal, 1 mean aggressive
        if MOT_show_flag:
            for id in bbox.keys():
                draw_color_idx = 0
                x1, y1, offset_x, offset_y = bbox[id]
                x2, y2 = x1 + offset_x, y1 + offset_y
                if BC_result_flag and id in self.BC.result.keys():
                    bcr = self.BC.result[id]
                    if bcr == -1:
                        draw_color_idx = 1
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), self.BC.bbox_color[draw_color_idx], 2)
                cv2.putText(frame, str(id), (int(x1), int(y1)), cv2.FONT_HERSHEY_PLAIN,\
                        1, (255, 255, 0), thickness=1)

        # Draw trajectory using point
        traj_show_flag = True
        try:
            # # Draw past traj
            # if traj_show_flag:
            #     for f in self.TP.traj:
            #         for k, v in f.items():
            #             cv2.circle(frame, (int(v[0]), int(v[1])), 3, (0,0,255), -1)
            # # Draw future traj
            # if self.TP.future_trajs is not None and traj_show_flag:
            #     for k, v in self.TP.future_trajs.items():
            #         for x, y in v:
            #             cv2.circle(frame, (int(x), int(y)), 3, (255,0,0), -1)
            if t_total is not None:
                fps = 1 / (time.time() - t_total)
                self.avg_fps += fps
                cv2.putText(frame, "FPS: "+str(fps), (0, 40),cv2.FONT_HERSHEY_PLAIN,\
                        2, (255,255,255), thickness=3)
            # wk: if future_traj is drawn, then waitkey set 0 for better visualization.
            wk = 0 if future_traj_flag else 1
            # cv2.imwrite("tmp_img/{}.jpg".format(self.f_id), frame)
            # self.writer.write(frame)
            cv2.imshow('t', frame)
            wk = 1
            if cv2.waitKey(wk) == 27: # whether is pressed ESC key.
                print("ESC pressed.")
                return True
            self.f_id += 1
        except Exception as e:
            print("Exception is happened: {}".format(e))
