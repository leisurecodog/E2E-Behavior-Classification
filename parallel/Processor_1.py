from system_util import ID_check
# def run(frame_dict, objdet_dict, MOT_dict, Future_traj_dict, BC_dict, lock):
def run(*params):
    frame_dict, objdet_dict, MOT_dict, Future_traj_dict, BC_dict, lock = params
    from MOT_module.MOT import MOT
    from TP_module.TP import TP
    from BC_module.BC import BC
    module_MOT = MOT()
    module_TP = TP()
    module_BC = BC(module_TP.traj_len_required)
    frame_id = 0
    
    
    current_traj_dict = {}
    # future_traj_dict = {}
    reset_flag = False
    ID_check(MOT_dict, 'MOT_dict')
    # print("ID", id(MOT_dict))
    # ==================================================
    while True:
        # lock.acquire()
        if frame_id in frame_dict:
            # lock.release()
            data = frame_dict[module_MOT.frame_id]
            module_MOT.run(data, objdet_dict, MOT_dict, lock)
            # print("MOT done")
            data = MOT_dict[frame_id]

            # ==============================================

            module_TP.update_traj(data, frame_id, current_traj_dict)
            # print("TP update Trajectory from MOT.")
            if module_TP.is_some_id_predictable():
                module_TP.run(frame_id, current_traj_dict, Future_traj_dict)
                # print("TP done")
            
            # ==============================================
            # print(module_TP.ID_counter)
            if module_BC.is_satisfacation(module_TP.ID_counter):
                reset_flag = True
                # print("???????????????", list(Future_traj_dict.keys())[-1])
                module_BC.run(current_traj_dict.values(), Future_traj_dict[frame_id])
                BC_dict[frame_id] = module_BC.result
                # print("BC done")
            if reset_flag:
                module_TP.traj_reset()
                reset_flag = False
            frame_id += 1
            # print(frame_id)
        # else:
            # lock.release()
    # ===================================================================