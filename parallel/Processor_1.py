from system_util import ID_check
import time

# def run(frame_dict, objdet_dict, MOT_dict, Future_traj_dict, BC_dict, lock):
def run(*params):
    frame_dict, objdet_dict, MOT_dict, current_traj_id_dict, Future_traj_dict, BC_dict, lock, reset_flag = params
    from MOT_module.MOT import MOT
    from TP_module.TP import TP
    from BC_module.BC import BC
    module_MOT = MOT()
    module_TP = TP()
    module_BC = BC(module_TP.traj_len_required)
    frame_id = 0
    current_traj_frame_list = list()

    maintain_history_traj_len = 30
    execute_freq = 30
    # ==================================================
    while True:
        # lock.acquire()
        if frame_id in frame_dict:
            # lock.release()
            data = frame_dict[module_MOT.frame_id]
            module_MOT.run(data, objdet_dict, MOT_dict, lock)
            print("MOT done")
            data = module_MOT.current_MOT
            
            # ======================== Update current trajectory Buffer ======================
            module_TP.update_traj(data)
            # current_traj_frame_dict[frame_id] = module_TP.current_frame
            current_traj_frame_list.append(module_TP.current_frame)
            # print(len(current_traj_frame_list))

            for k, v in module_TP.current_frame.items():
                if k not in current_traj_id_dict:
                    current_traj_id_dict[k] = []
                tmp_list = current_traj_id_dict[k]
                tmp_list.append(v)
                current_traj_id_dict[k] = tmp_list
                # module_TP.ID_counter[k] = len(current_traj_id_dict[k])
            
            if frame_id % execute_freq == 0:
                # update current_traj_frame_list buffer
                # Cutting traj
                if len(current_traj_frame_list) >= maintain_history_traj_len:
                    current_traj_frame_list = current_traj_frame_list[-maintain_history_traj_len:]
                
                # ======================== update total traj ======================
                tmp_dict = {}
                for frame_data in current_traj_frame_list:
                    for k, v in frame_data.items():
                        if k not in tmp_dict:
                            tmp_dict[k] = []
                        tmp_dict[k].append(v)
                
                module_TP.update_ID_counter(tmp_dict)
                current_traj_id_dict.update(dict(tmp_dict))
                
                # ======================== TP Working ======================
                if module_TP.is_some_id_predictable():
                    T1 = time.time()
                    module_TP.run(current_traj_id_dict, lock)
                    print("TP done")
                    
                Future_traj_dict[frame_id] = module_TP.result
                module_TP.result = None
                # ======================== BC Working ======================
                if module_BC.is_satisfacation(module_TP.ID_counter) and frame_id % execute_freq == 0:
                    module_BC.run(current_traj_frame_list, Future_traj_dict[frame_id])
                    print("BC done")

                BC_dict[frame_id] = module_BC.result
                module_BC.result = None

            else:
                Future_traj_dict[frame_id] = None
                BC_dict[frame_id] = None
            
            frame_id += 1