
# from torch.multiprocessing import Process, Manager
from multiprocessing.connection import wait
import torch.multiprocessing as torch_mp

from os import system
import cv2
from system_UI import MyDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QGridLayout, QLabel, QPushButton, QWidget
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
import time
import numpy as np
from system_util import ID_check

g_frame = np.zeros((720, 640, 3))

def run():
    '''
    main Process: execute OT module
    '''
    # from MOT_module.MOT_processor import run as MOT_run
    # from TP_module.TP_processor import run as TP_run
    from OT_module.OT import OT
    from Processor_1 import run as P1_run

    
    sys_args = system_parser.get_parser()
    module_OT = OT()
    
    frame_id = 0
    manager = torch_mp.Manager()
    dict_frame = manager.dict()
    dict_objdet = manager.dict()
    dict_MOT = manager.dict()
    dict_traj_current = manager.dict()
    dict_traj_future = manager.dict()
    dict_BC = manager.dict()
    dict_Flag = manager.dict()
    lock = torch_mp.Lock()
    p_list = [[]] * 20
    p_list[0] = torch_mp.Process(target=P1_run, args=(dict_frame, dict_objdet, dict_MOT, dict_traj_future, dict_BC, lock,))
    p_list[0].start()
    p_list[1] = torch_mp.Process(target=Input_reader, args=(sys_args, dict_frame,))
    p_list[1].start()
    p_list[2] = torch_mp.Process(target=Output_reader, args=(dict_frame, dict_MOT, \
        dict_traj_current, dict_traj_future,lock,))
    p_list[2].start()
    # ID_check(dict_MOT, "dict_MOT")
    while True:
      # ret_val: True -> Read image, False -> No image
      # frame: image frame.
        
      # start working when have image.

            # bounding box and ID infomation
        if frame_id in dict_frame:
            frame = dict_frame[frame_id]
            dict_frame[frame_id] = frame
            module_OT.run(frame, frame_id, dict_objdet)
            frame_id += 1

def window(g_frame):
    # global g_frame
    
    import sys
    app = QApplication(sys.argv)
    dialog = MyDialog()
    dialog.show()
    dialog.set_img(g_frame)
    sys.exit(app.exec_())

def Input_reader(sys_args, dict_frame):
    global g_frame
    cap = cv2.VideoCapture(sys_args.video_path)
    frame_id = 0
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            g_frame = frame
            if sys_args.resize:
                frame = cv2.resize(frame, (sys_args.size))
            dict_frame[frame_id] = frame
            frame_id += 1
    

def Output_reader(dict_frame, dict_MOT, list_traj_current, dict_traj_future, lock):    
    frame_id = 0
    last_frame_time = 0
    while True:
        # output: show video if all data is in share dictionary.
        if frame_id in dict_MOT:
            entry_time = time.time()
            if frame_id != 0:
                waiting_time = entry_time - last_frame_time
                print("Waiting time: {}, FPS: {}".format(waiting_time, 1/waiting_time))
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
            last_frame_time = entry_time

# def show(self, frame, t_total=None):
#         bbox = self.MOT.result
#         MOT_show_flag = False
#         future_traj_flag = True if self.TP.future_trajs is not None else False
#         BC_result_flag = True if self.BC.result is not None else False

#         # Draw bounding box & agent ID and Behavior Classification Result
#         # draw_color_idx: 0 mean normal, 1 mean aggressive
#         if MOT_show_flag:
#             for id in bbox.keys():
#                 draw_color_idx = 0
#                 x1, y1, offset_x, offset_y = bbox[id]
#                 x2, y2 = x1 + offset_x, y1 + offset_y
#                 if BC_result_flag and id in self.BC.result.keys():
#                     bcr = self.BC.result[id]
#                     if bcr == -1:
#                         draw_color_idx = 1
#                 cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), self.BC.bbox_color[draw_color_idx], 2)
#                 cv2.putText(frame, str(id), (int(x1), int(y1)), cv2.FONT_HERSHEY_PLAIN,\
#                         1, (255, 255, 0), thickness=1)

#         # Draw trajectory using point
#         traj_show_flag = True
#         try:
#             # Draw past traj
#             if traj_show_flag:
#                 for f in self.TP.traj:
#                     for k, v in f.items():
#                         cv2.circle(frame, (int(v[0]), int(v[1])), 3, (0,0,255), -1)
#             # Draw future traj
#             if self.TP.future_trajs is not None and traj_show_flag:
#                 for k, v in self.TP.future_trajs.items():
#                     for x, y in v:
#                         cv2.circle(frame, (int(x), int(y)), 3, (255,0,0), -1)
#             if t_total is not None:
#                 cv2.putText(frame, "FPS: "+str(1.0/(time.time()-t_total)), (0, 20),cv2.FONT_HERSHEY_PLAIN,\
#                         1, (255,255,255), thickness=1)
#             # wk: if future_traj is drawn, then waitkey set 0 for better visualization.
#             wk = 0 if future_traj_flag else 1
#             self.writer.write(frame)
#             cv2.imshow('t', frame)
#             wk = 1
#             if cv2.waitKey(wk) == 27: # whether is pressed ESC key.
#                 print("ESC pressed.")
#                 return True
#         except Exception as e:
#             print("Exception is happened: {}".format(e))

if __name__ == '__main__':
    torch_mp.set_start_method('spawn')
    import system_parser
    # import threading
    # t1 = threading.Thread(target=window, args=(g_frame,))
    # t1.start()
    run()
