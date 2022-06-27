from system_util import ID_check
import time


def run(*params):
    frame_dict, objdet_dict, BC_dict = params
    from MOT_module.MOT import MOT
    from TP_module.TP import TP
    from BC_module.BC import BC
    module_MOT = MOT()
    module_TP = TP()
    module_BC = BC(module_TP.traj_len_required)
    frame_id = 0
    execute_freq = 1
    show_msg_flag = False
    # read test
    # import cv2
    # import system_parser
    # sys_args = sys_args = system_parser.get_parser()
    # cap = cv2.VideoCapture(sys_args.video_path)

    # ==================================================z
    while True:
        # read test
        # ret_val, frame = cap.read()
        # if ret_val:
        #     if sys_args.resize:
        #         frame = cv2.resize(frame, (sys_args.size))
        #     frame_dict.update({frame_id:frame})
        # read test end =================================================

        if frame_id in frame_dict:
            frame = frame_dict[module_MOT.frame_id]
            t1 = time.time()
            module_MOT.run(frame, objdet_dict)
            if show_msg_flag:
                print("MOT done ", time.time() - t1)
            # MOT_dict.update({frame_id:module_MOT.current_MOT})
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
            else:
                print("?")
                # Future_traj_dict.update({frame_id:None})
                BC_dict.update({frame_id:None})
                # BC_dict[frame_id] = None
            frame_id += 1