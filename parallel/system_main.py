
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
    from Processor_2 import run as Input_reader
    from Processor_3 import run as Output_reader
    sys_args = system_parser.get_parser()
    module_OT = OT()
    
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
    # reset_flag = manager.Value(bool, False)
    lock = torch_mp.Lock()
    # share data config =========================================
    
    # create subprocess
    p_list = [[]] * 3
    p_list[0] = torch_mp.Process(target=P1_run,
    args=(dict_frame, dict_objdet, 
        dict_MOT, dict_traj_id_dict, 
        dict_traj_future, dict_BC, 
        lock,))
    p_list[1] = torch_mp.Process(target=Input_reader, args=(sys_args, dict_frame,))
    p_list[2] = torch_mp.Process(target=Output_reader, args=(dict_frame, dict_MOT, \
        dict_traj_id_dict, dict_traj_future, dict_BC, lock,))
    # start each subprocess
    p_list[0].start()
    p_list[1].start()        
    p_list[2].start()
    frame_id = 0

    while True:
        # waiting frame and object detection result.
        if frame_id in dict_frame and frame_id in dict_objdet:
            frame = dict_frame[frame_id]
            # dict_frame[frame_id] = frame
            module_OT.run(frame, frame_id, dict_objdet)
            # save OT result to share dict.
            dict_OT[frame_id] = module_OT.OTS.msg
            frame_id += 1

def window(g_frame):
    # global g_frame
    
    import sys
    app = QApplication(sys.argv)
    dialog = MyDialog()
    dialog.show()
    dialog.set_img(g_frame)
    sys.exit(app.exec_())

if __name__ == '__main__':
    torch_mp.set_start_method('spawn')
    import system_parser
    run()
