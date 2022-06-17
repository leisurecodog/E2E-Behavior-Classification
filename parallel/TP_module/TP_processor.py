from TP_module.util import prGreen
import time
import numpy as np
import threading

def run(MOT_dict, current_traj_list, future_traj_dict):
    from TP_module.TP import TP
    module = TP()
    frame_id = 0
    # ===================================================================
    while True:
        if not MOT_dict[frame_id] is not None:
            data = MOT_dict[frame_id]
            # print(id(current_traj_list))
            module.update_traj(data, current_traj_list)
            print("TP update Trajectory from MOT.")
            module.run(current_traj_list, future_traj_dict)
    # ===================================================================
        
