
from multiprocessing.connection import wait
import torch.multiprocessing as torch_mp
# from os import system
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
    from OT_module.OT import OT
    from Processor_1 import run as P1_run

    
    sys_args = system_parser.get_parser()
    module_OT = OT()
    
    frame_id = 0
    # share data config =========================================
    manager = torch_mp.Manager()
    dict_frame = manager.dict() # save frame
    dict_objdet = manager.dict() # save objdet result
    dict_MOT = manager.dict() # save MOT result
    dict_traj_frame_current = manager.dict() # save traj by format frame -> internal for P1
    dict_traj_id_dict = manager.dict() # save traj by format {id : traj}
    dict_traj_future = manager.dict()
    dict_BC = manager.dict()
    dict_OT = manager.dict()
    reset_flag = manager.Value(bool, False)
    lock = torch_mp.Lock()
    # share data config =========================================

    p_list = [[]] * 3
    p_list[0] = torch_mp.Process(target=P1_run,
    args=(dict_frame, dict_objdet, 
        dict_MOT, dict_traj_id_dict, 
        dict_traj_future, dict_BC, 
        lock, reset_flag,))
    p_list[1] = torch_mp.Process(target=Input_reader, args=(sys_args, dict_frame,))
    p_list[2] = torch_mp.Process(target=Output_reader, args=(dict_frame, dict_MOT, \
        dict_traj_id_dict, dict_traj_future, dict_BC, lock, reset_flag))
    
    p_list[0].start()
    p_list[1].start()        
    p_list[2].start()
    # ID_check(dict_MOT, "dict_MOT")
    while True:
      
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
    '''
    function of Input_reader:
        Description:
            Read frame from video and put each frame into dict_frame container.
        Input:
            sys_args: All args using for system is there, more detail arguments please see system_parser.py
            dict_frame: A dictionary from torch.mp.manager.dict, it responses for storing frame.
        Output:
            None
    '''
    global g_frame
    cap = cv2.VideoCapture(sys_args.video_path)
    frame_id = 0
    while True:
        '''
        ret_val: { True -> Read image, False -> No image }
        frame: image frame.
        '''
        ret_val, frame = cap.read()
        if ret_val:
            g_frame = frame
            if sys_args.resize:
                frame = cv2.resize(frame, (sys_args.size))
            dict_frame[frame_id] = frame
            frame_id += 1
    
def Output_reader(dict_frame, dict_MOT, dict_traj_current, dict_traj_future, dict_BC, lock, reset_flag):    
    frame_id = 0
    last_frame_time = 0
    while True:
        FPS = 0
        # output: show video if all data is in share dictionary.
        # if frame_id in dict_MOT and frame_id in dict_traj_current and frame_id in dict_traj_future:
        if frame_id in dict_MOT and frame_id in dict_BC:
            entry_time = time.time()
            if frame_id != 0:
                waiting_time = entry_time - last_frame_time
                FPS = 1 / waiting_time
                print("FPS: {}".format(FPS))
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
            for k, traj in dict_traj_current.items():
                for v in traj:
                    cv2.circle(fm, (int(v[0]), int(v[1])), 3, (0,0,255), -1)
            
            cv2.putText(fm, "FPS: {}".format(FPS), (0, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), thickness=1)

            cv2.imshow('t', fm)
            wk = 1
            if cv2.waitKey(wk) == 27:
                break
            frame_id += 1
            last_frame_time = entry_time

if __name__ == '__main__':
    torch_mp.set_start_method('spawn')
    import system_parser
    run()
