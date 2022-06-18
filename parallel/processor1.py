

def run(frame_dict, objdet_dict, res_traj_dict):
    from MOT_module.MOT import MOT
    from TP_module.TP import TP
    module_MOT = MOT()
    module_TP = TP()
    
    frame_id = 0
    current_traj_dict = {}
    # ==================================================
    while True:
        if frame_dict[frame_id] is not None:
            data = frame_dict[frame_id]
            module_MOT.run(data, objdet_dict)

    # ===================================================================
        
            data = module_MOT.results
            # print(id(current_traj_list))
            module_TP.update_traj(data, frame_id, current_traj_dict)
            print("TP update Trajectory from MOT.")
            module_TP.run(current_traj_dict.values(), res_traj_dict)
            frame_id += 1
    # ===================================================================
        