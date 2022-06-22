from system_util import ID_check
import time

# def run(frame_dict, objdet_dict, MOT_dict, Future_traj_dict, BC_dict, lock):
def run(*params):
    frame_dict, objdet_dict, MOT_dict, current_traj_id_dict, Future_traj_dict, BC_dict, lock = params
    from MOT_module.MOT import MOT
    from TP_module.TP import TP
    from BC_module.BC import BC
    module_MOT = MOT()
    module_TP = TP()
    module_BC = BC(module_TP.traj_len_required)
    frame_id = 0
    execute_freq = 1
    # ==================================================
    while True:
        # lock.acquire()
        if frame_id in frame_dict:
            # lock.release()
            data = frame_dict[module_MOT.frame_id]
            module_MOT.run(data, objdet_dict, MOT_dict, lock)
            # print("MOT done")
            data = module_MOT.current_MOT.copy()
            
            # ======================== Update current trajectory Buffer ======================
            module_TP.update_traj(data)
            if frame_id % execute_freq == 0:
                current_traj_id_dict.clear()
                current_traj_id_dict.update(module_TP.traj_id_dict)
                
                # ======================== TP Working ======================
                if module_TP.is_some_id_predictable():
                    t1 = time.time()
                    module_TP.run()
                    # print("TP done ", time.time() - t1)
                # print(module_TP.result)
                Future_traj_dict[frame_id] = module_TP.result
                
                # ======================== BC Working ======================
                if module_BC.is_satisfacation(module_TP.current_frame_id):
                    module_BC.run(module_TP.traj, Future_traj_dict[frame_id])
                    print("BC done")
                BC_dict[frame_id] = module_BC.result
            else:
                Future_traj_dict[frame_id] = None
                BC_dict[frame_id] = None
            # Future_traj_dict[frame_id] = None
            # BC_dict[frame_id] = None
            frame_id += 1