import time
import cv2
from numpy import average
import numpy as np
import sys

def run(*params):
    dict_frame, dict_MOT, dict_future, dict_BC,\
    dict_OT, lock, dict_config, signal, qt_set_img, qt_set_fps = params
    frame_id = 0
    entry_time = 0
    total_fps = 0
    counter = 0
    bbox = []
    current_id = []
    history_traj = {}
    colors = [(255, 255, 255), (0, 0, 255)]
    bc_happen_counter = 0
    bc_happen_limit = 10
    bcr = None
    bc_none_flag = False
    FPS = 0
    OT_msg = ''
    while True:
        if dict_config['Exit'] :
            print("EXIT")
            # qt_stop_func()
            return
        # output: show video if all data is in share dictionary.
        # 等待超車輔助結果
        while dict_config['OT'] and frame_id not in dict_OT:
            continue

        # 等待行為分類結果
        while frame_id not in dict_BC:
            continue
        # handle bc result.
        if dict_BC[frame_id] is not None:
            bc_happen_counter += 1
            bc_none_flag = False
            if bc_happen_counter == bc_happen_limit:
                bc_happen_counter = 0
                bcr = dict_BC[frame_id]
        else:
            bc_none_flag = True

        if dict_config['OT']:
            if bc_none_flag:
                OT_msg = dict_OT[frame_id]
            else:
                OT_msg = ''
        
        fm = dict_frame[frame_id]
        mot_exist_flag = dict_MOT[frame_id] is not None
        if mot_exist_flag:
            bbox = dict_MOT[frame_id]
            limit = 5
            # 畫出物件框
            for ID, current in bbox.items():
                if ID not in history_traj:
                    history_traj[ID] = []
                x1, y1, offset_x, offset_y = current
                x2, y2 = x1 + offset_x , y1 + offset_y
                if dict_config['MOT']:
                    if dict_config['ID'] == 0 or dict_config['ID'] == ID:
                        color = colors[0]
                        # if limit > 0 and bcr is not None and ID in bcr and not bc_none_flag:
                        if limit > 0 and bcr is not None and ID in bcr:
                            color = colors[1]
                            limit -= 1
                        cv2.rectangle(fm, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        cv2.putText(fm, str(ID), (int(x1), int(y1)), cv2.FONT_HERSHEY_PLAIN,\
                                3, (255, 255, 0), thickness=2)
                # record mot date each timestamp.
                history_traj[ID].append([x1 + offset_x // 2, y1 + offset_y // 2])
            # draw history trajectory.
            if dict_config['HTP']:
                current_id = dict_MOT[frame_id].keys()
                for ID, traj in history_traj.items():
                    if ID in current_id and (dict_config['ID'] == 0 or dict_config['ID'] == ID):
                        for v in traj[-30:]:
                            cv2.circle(fm, (int(v[0]), int(v[1])), 3, (0, 0, 255), -1)
                            
            # draw future trajectory
            if dict_config['FTP'] and frame_id in dict_future:
                current_id = dict_MOT[frame_id].keys()
                future = dict_future[frame_id]
                for ID, traj in future.items():
                    if ID in current_id and (dict_config['ID'] == 0 or dict_config['ID'] == ID):
                        for v in traj:
                            cv2.circle(fm, (int(v[0]), int(v[1])), 3, (255, 0, 0), -1)
        if entry_time != 0:
            ts = time.time() - entry_time
            FPS = 1 / ts
            qt_set_fps(FPS)
            total_fps += FPS
            counter += 1
            # qt_set_fps(total_fps/counter)
        entry_time = time.time()
        cv2.putText(fm, OT_msg, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        qt_set_img(fm)
        # input()
        frame_id += 1
    