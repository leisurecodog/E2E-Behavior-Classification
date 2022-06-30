
# from multiprocessing.connection import wait
import torch.multiprocessing as torch_mp
import torch
import threading
# from os import system
import cv2
import sys
import time
import numpy as np
from system_util import ID_check

#
from PyQt5 import QtWidgets, QtCore
from system_UI import MainWindow_controller
app = QtWidgets.QApplication(sys.argv)
UI_window = MainWindow_controller()
#

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
    entry_time = 0
    total_fps = 0
    counter = 0
    from Processor_1 import run as P1_run, run2
    from Processor_2 import run as Input_reader
    from Processor_3 import run as Output_reader
    # from p4 import p4
    module_OT = OT()
    # create subprocess
    p_list = [[]] * 3
    p_list[0] = torch_mp.Process(target=P1_run,
    # args=(dict_UI, dict_frame, dict_objdet, dict_BC, dict_MOT, dict_OT))
    args=(dict_frame, dict_objdet, dict_BC, dict_MOT, dict_OT))
    p_list[1] = torch_mp.Process(target=Input_reader, args=(dict_UI, dict_frame,))

    p_list[2] = torch_mp.Process(target=Output_reader, args=(dict_frame, dict_BC, dict_OT, dict_MOT,)) # dict_MOT, 
    
    # p_list[2] = torch_mp.Process(target=p4, args=(dict_frame,dict_objdet,dict_OT,))
    
    # start each subprocess
    for i in range(3):
        p_list[i].start()
        
    # while dict_UI['start'] == False:
    #     continue
    
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
            
            # # print(frame_id)
            # print("{} OT done:".format(frame_id), time.time() - t1)
            # while (frame_id not in dict_MOT) or (frame_id not in dict_BC):
            #     continue
            # if entry_time != 0:
            #     ts = time.time() - entry_time
            #     FPS = 1 / ts
            #     total_fps += FPS
            #     counter += 1
            #     print(total_fps / counter)
            # entry_time = time.time()
                # UI_window.set_img(frame)
            frame_id += 1
    
    for i in range(3):
        p_list[i].join()


if __name__ == '__main__':
    torch_mp.set_start_method('spawn')
    torch.set_num_threads(1)    
    # run()

    UI_window.show()
    sys.exit(app.exec_())