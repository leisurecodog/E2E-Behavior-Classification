
from multiprocessing.connection import wait
# from sys import last_traceback
import torch.multiprocessing as torch_mp
import threading
# from os import system
import cv2
# from system_UI import MyDialog
# from PyQt5.QtGui import QImage, QPixmap
# from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QGridLayout, QLabel, QPushButton, QWidget
# from PyQt5.QtGui import QIcon
# from PyQt5.QtCore import pyqtSlot
import time
import numpy as np
from system_util import ID_check


def run():
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
    dict_UI = manager.dict() # save frame
    dict_UI['start'] = False
    dict_UI['TP'] = False
    dict_UI['stop'] = False
    # share memory config =========================================
    from OT_module.OT import OT
    from Processor_1 import run as P1_run, run2
    from Processor_2 import run as Input_reader
    from Processor_3 import run as Output_reader
    
    
    
    # create subprocess
    p_list = [[]] * 3
    p_list[0] = torch_mp.Process(target=run2,
    args=(dict_UI, dict_frame, dict_objdet, dict_BC, dict_MOT, dict_OT))
        # dict_MOT, dict_traj_id_dict, 
        # dict_traj_future, dict_BC,))
    # p_list[2] = torch_mp.Process(target=Output_reader, args=(dict_frame, dict_BC, dict_OT, dict_MOT,)) # dict_MOT, 
    #     dict_traj_id_dict, dict_traj_future, 
    #     dict_BC,))
    p_list[1] = torch_mp.Process(target=Input_reader, args=(dict_UI, dict_frame,))

    # start each subprocess
    for i in range(2):
        p_list[i].start()
        
    while dict_UI['start'] == False:
        continue
    module_OT = OT()
    frame_id = 0
    while True:
        if frame_id in dict_objdet:
            frame = dict_frame[frame_id]
            t1 = time.time()
            module_OT.run(frame, dict_objdet[frame_id])
            # dict_OT[frame_id] = module_OT.OTS.msg
            # g_frame = frame.copy()
            # UI_window.setup_control(frame)
            dict_OT.update({frame_id:module_OT.OTS.msg})
            frame_id += 1
    
    for i in range(3):
        p_list[i].join()


if __name__ == '__main__':
    import torch
    # print(torch.get_num_threads())
    torch.set_num_threads(1)
    torch_mp.set_start_method('spawn')
    run()