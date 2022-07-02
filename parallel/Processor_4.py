import time
import cv2
from numpy import average
import numpy as np
import sys

def run(dict_frame, dict_MOT, dict_BC, dict_OT, dict_config, qt_set_img):
    frame_id = 0
    entry_time = 0
    total_fps = 0
    counter = 0
    bbox = []
    history_traj = {}
    FPS = 0
    while True:
        # print("Frame id {}".format(frame_id), event.is_set())
        if dict_config['Exit']:
            print("EXIT")
            return
        # output: show video if all data is in share dictionary.
        # if (frame_id in dict_OT) and (frame_id in dict_BC):
        while frame_id not in dict_OT:
            continue
        while frame_id not in dict_BC:
            continue
        
        fm = dict_frame[frame_id]
        mot_exist_flag = dict_MOT[frame_id] is not None
        # if dict_config['MOT'].is_set():
        if mot_exist_flag:
            bbox = dict_MOT[frame_id]
            for ID, current in bbox.items():
                if ID not in history_traj:
                    history_traj[ID] = []
                x1, y1, offset_x, offset_y = current
                x2, y2 = x1 + offset_x , y1 + offset_y
                if dict_config['MOT']:
                    if dict_config['ID'] == 0:
                        cv2.rectangle(fm, (int(x1), int(y1)), (int(x2), int(y2)), (255,255,255), 2)
                        cv2.putText(fm, str(ID), (int(x1), int(y1)), cv2.FONT_HERSHEY_PLAIN,\
                                3, (255, 255, 0), thickness=2)
                    elif dict_config['ID'] == ID:
                        cv2.rectangle(fm, (int(x1), int(y1)), (int(x2), int(y2)), (255,255,255), 2)
                        cv2.putText(fm, str(ID), (int(x1), int(y1)), cv2.FONT_HERSHEY_PLAIN,\
                                3, (255, 255, 0), thickness=2)
                history_traj[ID].append([x1 + offset_x // 2, y1 + offset_y // 2])

            if dict_config['HTP']:
                current_id = dict_MOT[frame_id].keys()
                for ID, traj in history_traj.items():
                    if ID in current_id:
                        for v in traj[-30:]:
                            cv2.circle(fm, (int(v[0]), int(v[1])), 3, (0, 0, 255), -1)
        
        

        # if entry_time != 0 and frame_id % 100 == 0:
        #     ts = time.time() - entry_time
        #     FPS = 1 / ts
        #     total_fps += FPS
        #     counter += 1
        #     print("1. Frame: {} \t FPS: {} \t Average FPS: {}"\
        #         .format( frame_id, ts, total_fps / counter))
        # entry_time = time.time()
        
        # lock.release()
                
        # if BC_result_flag and id in self.BC.result.keys():
        #     bcr = self.BC.result[id]
        #     if bcr == -1:
        #         draw_color_idx = 1
        # cv2.rectangle(fm, (int(x1), int(y1)), (int(x2), int(y2)), self.BC.bbox_color[draw_color_idx], 2)
        # draw trajs
        # print(frame_id, dict_traj_current)

        # for ID, traj in history_traj.items():
        #     # display traj when ID in current frame
        #     if ID not in bbox:
        #         continue
        #     for v in traj:
        #         cv2.circle(fm, (int(v[0]), int(v[1])), 3, (0,0,255), -1)
        
        # cv2.putText(fm, "average FPS: {}".format(FPS), (0, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), thickness=3)
        qt_set_img(fm)
        frame_id += 1
    