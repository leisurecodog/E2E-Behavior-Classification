from system_util import ID_check
import time
import threading
import cv2
import sys

def run(*params):
    frame_dict, objdet_dict, Future_traj_dict, BC_dict, MOT_dict, config_dict, lock, signal = params

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
            
            t1 = time.time()
            module_MOT.run(frame, objdet_dict, lock)
            if show_msg_flag:
                # print("MOT done ", time.time() - t1, frame_id)
                print("MOT done \t {}".format(frame_id))
            lock.acquire()    
            MOT_dict.update({frame_id:module_MOT.current_MOT})
            lock.release()
            
            data = module_MOT.current_MOT.copy() if module_MOT.current_MOT is not None else None
            if data is None:
                continue            
            # ======================== Update current trajectory Buffer ======================
            # current_traj_id_dict.clear()
            # current_traj_id_dict.update(module_TP.traj_id_dict)
            module_TP.update_traj(data)
            
            # ======================== TP Working ======================
            if config_dict['TP'] and module_TP.is_some_id_predictable():
                t1 = time.time()
                module_TP.run()
                if show_msg_flag:
                    # print("TP done ", time.time() - t1, frame_id)
                    print("TP done \t {}".format(frame_id))
                lock.acquire()
                Future_traj_dict.update({frame_id:module_TP.result})
                lock.release()
            # ======================== BC Working ======================
            if config_dict['BC'] and module_BC.is_satisfacation(module_TP.current_frame_ID_counter):
                t1 = time.time()
                module_BC.run(module_TP.traj, module_TP.result)
                if show_msg_flag:
                    # print("BC done ", time.time() - t1, frame_id)
                    print("BC done \t {}".format(frame_id))
            # lock.acquire()
            BC_dict.update({frame_id:module_BC.result})
            # lock.release()
            frame_id += 1
            # if signal.value == frame_id:
            #     print("mot module average time:", module_MOT.exe_time / module_MOT.counter)
            #     print("TP module average time:", module_TP.exe_time / module_TP.counter)
            #     print("bc module average time:", module_BC.exe_time / module_BC.counter)
            #     return
    