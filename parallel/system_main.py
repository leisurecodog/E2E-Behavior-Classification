
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
import threading
g_frame = np.zeros((720, 640, 3))

def run():
    global g_frame
    '''
    main Process: execute OT module
    '''
    # share memory config =========================================
    manager = torch_mp.Manager()
    dict_frame = manager.dict() # save frame
    dict_objdet = manager.dict() # save objdet result
    dict_MOT = manager.dict() # save MOT result
    dict_traj_id_dict = manager.dict() # save traj by format {id : traj}
    dict_traj_future = manager.dict()
    dict_BC = manager.dict()
    dict_OT = manager.dict()
    # lock = torch_mp.Lock()
    
    
    # share memory config =========================================
    
    # dict_frame = dict() # save frame
    # dict_objdet = dict() # save objdet result
    # dict_MOT = dict() # save MOT result
    # dict_traj_id_dict = dict() # save traj by format {id : traj}
    # dict_traj_future = dict()
    # dict_BC = dict()
    # dict_OT = dict()
    # lock = False
    from OT_module.OT import OT
    from Processor_1 import run as P1_run
    from Processor_2 import run as Input_reader
    from Processor_3 import run as Output_reader
    sys_args = system_parser.get_parser()
    module_OT = OT()
    
    # create subprocess
    p_list = [[]] * 3
    p_list[0] = torch_mp.Process(target=P1_run,
    args=(dict_frame, dict_objdet, dict_BC,))
        # dict_MOT, dict_traj_id_dict, 
        # dict_traj_future, dict_BC,))
    p_list[1] = torch_mp.Process(target=Input_reader, args=(sys_args, dict_frame,))
    p_list[2] = torch_mp.Process(target=Output_reader, 
    args=(dict_frame, dict_BC, dict_BC,)) # dict_MOT, 
    #     dict_traj_id_dict, dict_traj_future, 
    #     dict_BC,))
    # p_list[2] = torch_mp.Process(target=window, args=(dict_frame,))
    # start each subprocess
    
    for i in range(3):
        p_list[i].start()

    frame_id = 0
    while True:
        # waiting frame and object detection result.
        if frame_id in dict_frame and frame_id in dict_objdet:
            frame = dict_frame[frame_id]
            t1 = time.time()
            module_OT.run(frame, dict_objdet[frame_id])
            # print("OT done ", time.time() - t1)
            # save OT result to share dict.
            # dict_OT[frame_id] = module_OT.OTS.msg
            dict_OT.update({frame_id:module_OT.OTS.msg})
            frame_id += 1
    
    for i in range(3):
        p_list[i].join()

def window(dict_frame):
    # global g_frame
    frame_id = 0
    import sys
    app = QApplication(sys.argv)
    dialog = MyDialog(dict_frame)
    dialog.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    import torch
    # print(torch.get_num_threads())
    torch.set_num_threads(2)
    torch_mp.set_start_method('spawn')
    import system_parser
    run()
    # window()
