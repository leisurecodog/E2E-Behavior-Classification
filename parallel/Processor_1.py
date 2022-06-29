from system_util import ID_check
import time
import threading
import cv2
from PyQt5 import QtWidgets, QtCore
from system_UI import MainWindow_controller
import sys


app = QtWidgets.QApplication(sys.argv)

UI_window = MainWindow_controller()



def run(*params):
    stop_event, UI_dict, frame_dict, objdet_dict, BC_dict, MOT_dict, OT_dict = params

    from MOT_module.MOT import MOT
    from TP_module.TP import TP
    from BC_module.BC import BC
    module_MOT = MOT()
    module_TP = TP()
    module_BC = BC(module_TP.traj_len_required)
    frame_id = 0
    execute_freq = 1
    show_msg_flag = False
    entry_time = 0
    total_fps = 0
    counter = 0
    # ==================================================
    while not stop_event.is_set():
        if frame_id in frame_dict:
            frame = frame_dict[module_MOT.frame_id]
            # print(frame.shape)
            t1 = time.time()
            module_MOT.run(frame, objdet_dict)
            if show_msg_flag:
                print("MOT done ", time.time() - t1)
            MOT_dict.update({frame_id:module_MOT.current_MOT})
            data = module_MOT.current_MOT.copy()
            # ======================== Update current trajectory Buffer ======================
            module_TP.update_traj(data)
            if frame_id % execute_freq == 0:
                # current_traj_id_dict.clear()
                # current_traj_id_dict.update(module_TP.traj_id_dict)
                # ======================== TP Working ======================
                if module_TP.is_some_id_predictable():
                    t1 = time.time()
                    module_TP.run()
                    if show_msg_flag:
                        print("TP done ", time.time() - t1)
                
                # Future_traj_dict.update({frame_id:module_TP.result})
                # ======================== BC Working ======================
                if module_BC.is_satisfacation(module_TP.current_frame_ID_counter):
                    t1 = time.time()
                    module_BC.run(module_TP.traj, module_TP.result)
                    if show_msg_flag:
                        print("BC done ", time.time() - t1)
                BC_dict.update({frame_id:module_BC.result})
                # BC_dict[frame_id] = module_BC.result
                while frame_id not in OT_dict:
                    continue
                if entry_time != 0:
                    ts = time.time() - entry_time
                    FPS = 1 / ts
                    total_fps += FPS
                    counter += 1
                    print(total_fps / counter)
                entry_time = time.time()
                UI_window.set_img(frame)
            else:
                # Future_traj_dict.update({frame_id:None})
                BC_dict.update({frame_id:None})
                # BC_dict[frame_id] = None
            
            frame_id += 1


def run2(*params):
    stop_event = threading.Event()
    UI_dict, frame_dict, objdet_dict, BC_dict, MOT_dict, OT_dict = params
    t1 = threading.Thread(target=run, args=(stop_event, UI_dict, frame_dict, objdet_dict, BC_dict, MOT_dict, OT_dict,))
    UI_window.set_component(UI_dict, t1, stop_event)
    UI_window.show()
    sys.exit(app.exec_())
        