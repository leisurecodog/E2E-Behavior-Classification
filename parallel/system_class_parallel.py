import os
import logging
import numpy as np
import cv2
from TP_module.util import prGreen
import torch
import time
import torch.multiprocessing as torch_mp
import system_parser
from system_util import ID_check

class system_parallel():

    def __init__(self):
        # from MOT_module.MOT import MOT
        # from TP_module.TP import TP
        # from BC_module.BC import BC
        # from OT_module.OT import OT

        self.sys_args = system_parser.get_parser()
        self.cap = cv2.VideoCapture(self.sys_args.video_path if self.sys_args.demo == "video" else self.sys_args.camid)
        self.frame_id = 0
        manager = torch_mp.Manager()
        self.dict_frame = manager.dict()
        self.dict_objdet = manager.dict()
        self.dict_MOT = manager.dict()
        self.dict_traj_current = manager.dict()
        self.dict_traj_future = manager.dict()
        self.dict_BC = manager.dict()
        self.dict_Flag = manager.dict()
        self.lock = torch_mp.Lock()
        self.p_list = [[]] * 20

    def run(self):
        from Processor_1 import run as P1_run
        self.p_list[0] = torch_mp.Process(target=P1_run, args=(self.dict_frame,\
             self.dict_objdet, self.dict_MOT, self.dict_traj_future, self.dict_BC, self.lock,))
        self.p_list[0].start()
        ID_check(self.dict_MOT, "dict_MOT")
        self.p_list[2] = torch_mp.Process(target=self.Display_run, args=(self.dict_frame, self.dict_MOT, \
            self.dict_traj_current, self.dict_traj_future, self.lock,))
        self.p_list[2].start()
        while True:
        # ret_val: True -> Read image, False -> No image
        # frame: image frame.
            ret_val, frame = self.cap.read()
        # start working when have image.
            if ret_val:
                g_frame = frame
                t_time_1 = time.time()            
                if self.sys_args.resize:
                    frame = cv2.resize(frame, (self.sys_args.size))

                # bounding box and ID infomation
                self.dict_frame[frame_id] = frame

                frame_id += 1
                # if sys.reset:
                #     sys.TP.traj_reset()
            else:
                print("video is end.")
                '''
                # break # this is wrong, if video is end, just mean that read is end,
                # but system still running. so call join for each process instead of 
                '''
                self.p_list[0].join()
                self.p_list[2].join()

    def Display_run(self, dict_frame, dict_MOT, list_traj_current, dict_traj_future, lock):
        frame_id = 0
        while True:
            # lock.acquire()
            if frame_id in dict_frame and frame_id in dict_MOT:
                bbox = dict_MOT[frame_id]
                # lock.release()
                fm = dict_frame[frame_id]
                for id in bbox.keys():
                    draw_color_idx = 0
                    x1, y1, offset_x, offset_y = bbox[id]
                    x2, y2 = x1 + offset_x, y1 + offset_y
                    # if BC_result_flag and id in self.BC.result.keys():
                    #     bcr = self.BC.result[id]
                    #     if bcr == -1:
                    #         draw_color_idx = 1
                    # cv2.rectangle(fm, (int(x1), int(y1)), (int(x2), int(y2)), self.BC.bbox_color[draw_color_idx], 2)
                    cv2.rectangle(fm, (int(x1), int(y1)), (int(x2), int(y2)), (255,255,255), 2)
                    cv2.putText(fm, str(id), (int(x1), int(y1)), cv2.FONT_HERSHEY_PLAIN,\
                            1, (255, 255, 0), thickness=1)
                cv2.imshow('t', fm)
                if cv2.waitKey(1) == 27:
                    break
                frame_id += 1
            # else:
            #     lock.release()
