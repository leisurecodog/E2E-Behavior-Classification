from system_util import ID_check
import time

# def run(frame_dict, objdet_dict, MOT_dict, Future_traj_dict, BC_dict, lock):
def run(*params):
    frame_dict, objdet_dict, MOT_dict, current_traj_frame_dict, current_traj_id_dict, Future_traj_dict, BC_dict, lock = params
    from MOT_module.MOT import MOT
    from TP_module.TP import TP
    from BC_module.BC import BC
    module_MOT = MOT()
    module_TP = TP()
    module_BC = BC(module_TP.traj_len_required)
    frame_id = 0

    # ==================================================
    while True:
        # lock.acquire()
        if frame_id in frame_dict:
            # lock.release()
            data = frame_dict[module_MOT.frame_id]
            module_MOT.run(data, objdet_dict, MOT_dict, lock)
            print("MOT done")
            data = MOT_dict[frame_id]
            # ==============================================
            # Update current trajectory Buffer
            module_TP.update_traj(data)
            current_traj_frame_dict[frame_id] = module_TP.current_frame
            
            for k, v in module_TP.current_frame.items():
                if k not in current_traj_id_dict:
                    current_traj_id_dict[k] = []
                tmp_list = current_traj_id_dict[k]
                tmp_list.append(v)
                current_traj_id_dict[k] = tmp_list
                module_TP.ID_counter[k] = len(current_traj_id_dict[k])
            
            # Update id_counter in TP module
            # ==============================================
            # print("TP update Trajectory from MOT.")
            # print(module_TP.ID_counter)
            if module_TP.is_some_id_predictable():
                T1 = time.time()
                module_TP.run(current_traj_id_dict)
                print("TP Time: ", time.time()-T1)
                # print("TP done")
                Future_traj_dict[frame_id] = (dict(zip(module_TP.ids, module_TP.future)))
            if frame_id not in Future_traj_dict:
                Future_traj_dict[frame_id] = "None"
            # ==============================================
            # print(module_TP.ID_counter)
            if module_BC.is_satisfacation(module_TP.ID_counter):
                # reset_flag = True
                # print("???????????????", list(Future_traj_dict.keys())[-1])
                module_BC.run(current_traj_frame_dict.values(), Future_traj_dict[frame_id])
                BC_dict[frame_id] = module_BC.result
                print("BC done")
                module_TP.traj_reset()
                current_traj_frame_dict.update(dict())
                # reset_flag = False
            if frame_id not in BC_dict:
                BC_dict[frame_id] = "None"
            frame_id += 1
            # print(frame_id)
        # else:
            # lock.release()
    # ===================================================================