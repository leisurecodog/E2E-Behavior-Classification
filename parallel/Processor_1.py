from system_util import ID_check
import time
import threading
import cv2
import sys

def run(*params):
    frame_dict, objdet_dict, Future_traj_dict, BC_dict, MOT_dict, config_dict = params

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
    FPS = 0
    # ==================================================
    while True:
        if frame_id in frame_dict:
            frame = frame_dict[module_MOT.frame_id]
            # print(frame.shape)
            t1 = time.time()
            module_MOT.run(frame, objdet_dict)
            if show_msg_flag:
                # print("MOT done ", time.time() - t1, frame_id)
                print("MOT done \t {}".format(frame_id))
            MOT_dict.update({frame_id:module_MOT.current_MOT})
            data = module_MOT.current_MOT.copy()
            # ======================== Update current trajectory Buffer ======================
            # current_traj_id_dict.clear()
            # current_traj_id_dict.update(module_TP.traj_id_dict)
            # print("TP module:", config_dict['TP'])
            
            if config_dict['TP']:
                module_TP.update_traj(data)
                # ======================== TP Working ======================
                if module_TP.is_some_id_predictable():
                    t1 = time.time()
                    module_TP.run()
                    if show_msg_flag:
                        # print("TP done ", time.time() - t1, frame_id)
                        print("TP done \t {}".format(frame_id))
            
                Future_traj_dict.update({frame_id:module_TP.result})
            # ======================== BC Working ======================
            if module_BC.is_satisfacation(module_TP.current_frame_ID_counter):
                t1 = time.time()
                module_BC.run(module_TP.traj, module_TP.result)
                if show_msg_flag:
                    # print("BC done ", time.time() - t1, frame_id)
                    print("BC done \t {}".format(frame_id))
            BC_dict.update({frame_id:module_BC.result})
            frame_id += 1