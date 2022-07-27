from system_util import ID_check
import time
import threading
import cv2
import sys

def run(*params):
    frame_dict, objdet_dict, Future_traj_dict, BC_dict, MOT_dict, config_dict, lock, signal = params
    # 參數說明：
	# 除config_dict外，*_dict皆為共享記憶體，
	# config_dict 主要儲存UI傳遞至行程內的資訊
	# config_dict['TP'], config_dict['BC']儲存 True/False
	# 代表是否執行模組
	# frame_dict: 儲存frame，每個行程都會用到
	# objdet_dict: 儲存物件偵測資訊，傳至行程2(Overtaking module)
	# Future_traj_dict: 儲存軌跡預測預測的軌跡，傳至行程4(demo用)
	# BC_dict: 行為分類結果，傳至行程4(沒有激進駕駛就會顯示Overtaking module的結果)
	# MOT_dict: 多物件追蹤結果，傳至行程4(demo用)

    from MOT_module.MOT import MOT
    from TP_module.TP import TP
    from BC_module.BC import BC
    module_MOT = MOT()
    module_TP = TP()
    module_BC = BC(module_TP.traj_len_required)
    frame_id = 0
    
    show_msg_flag = False
    FPS = 0
    # ==================================================
    while True:
        if frame_id in frame_dict:
            frame = frame_dict[module_MOT.frame_id]
            
            t1 = time.time()
            # ======================== MOT Work ======================
            module_MOT.run(frame, objdet_dict, lock)
            if show_msg_flag:
                print("MOT done ", time.time() - t1, frame_id)
            MOT_dict.update({frame_id:module_MOT.current_MOT})
            
            data = module_MOT.current_MOT.copy() if module_MOT.current_MOT is not None else None
            if data is None:
                continue            
            # ======================== Update current trajectory Buffer ======================
            module_TP.update_traj(data)
            
            # ======================== TP Work ======================
            if config_dict['TP'] and module_TP.is_some_id_predictable():
                t1 = time.time()
                module_TP.run()
                if show_msg_flag:
                    print("TP done ", time.time() - t1, frame_id)
                Future_traj_dict.update({frame_id:module_TP.result})
            # ======================== BC Work ======================
            if config_dict['BC'] and module_BC.is_satisfacation(module_TP.current_frame_ID_counter):
                t1 = time.time()
                module_BC.run(module_TP.traj, module_TP.result)
                if show_msg_flag:
                    print("BC done ", time.time() - t1, frame_id)
            BC_dict.update({frame_id:module_BC.result})

            frame_id += 1

    