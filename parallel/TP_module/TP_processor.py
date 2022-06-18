from TP_module.util import prGreen
import time
import numpy as np
import threading

def run(MOT_dict, current_traj_dict, future_traj_dict):
    from TP_module.TP import TP
    module = TP()
    frame_id = 0
    # ===================================================================
    while True:
        if frame_id in MOT_dict:
            data = MOT_dict[frame_id]
            # print(id(current_traj_list))
            module.update_traj(data, frame_id, current_traj_dict)
            print("TP update Trajectory from MOT.")
            module.run(frame_id, current_traj_dict, future_traj_dict)
            frame_id += 1
    # ===================================================================
        
