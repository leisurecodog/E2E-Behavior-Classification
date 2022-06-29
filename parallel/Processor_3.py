import time
import cv2
from numpy import average
import numpy as np
import threading
# from PyQt5 import QtWidgets, QtCore
# from system_UI import MainWindow_controller
import sys
# app = QtWidgets.QApplication(sys.argv)
# UI_window = MainWindow_controller()


def rd_func(dict_frame, dict_BC, dict_OT, dict_MOT):
    frame_id = 0
    last_frame_time = 0
    average_FPS = 0
    lowest_FPS = 100
    history_traj = {}
    FPS = 0
    while True:
        # output: show video if all data is in share dictionary.
        if (frame_id in dict_frame):
            # fm = dict_frame[frame_id]
            
            # UI_window.setup_control(fm)
            
            
            fm = dict_frame[frame_id]
            # update history_traj dict
            while frame_id not in dict_MOT:
                continue
            # bbox = dict_MOT[frame_id]
            # for ID, current in bbox.items():
            #     # if ID not in history_traj:
            #         # history_traj[ID] = []
            #     x1, y1, offset_x, offset_y = current
            #     x2, y2 = x1 + offset_x , y1 + offset_y
            #     cv2.rectangle(fm, (int(x1), int(y1)), (int(x2), int(y2)), (255,255,255), 2)
            #     cv2.putText(fm, str(ID), (int(x1), int(y1)), cv2.FONT_HERSHEY_PLAIN,\
            #             1, (255, 255, 0), thickness=1)
            
                # history_traj[ID].append([x1 + offset_x // 2, y1 + offset_y // 2])
            # while frame_id not in dict_BC:
            #     continue
            # while frame_id not in dict_OT:
            #     continue
            entry_time = time.time()
            if frame_id != 0:
                waiting_time = entry_time - last_frame_time
                FPS = 1 / waiting_time
                if frame_id > 20:
                    lowest_FPS = min(lowest_FPS, FPS)

                print("Frame: {} \t FPS: {} \t Average FPS: {}".format(frame_id, FPS, average_FPS/frame_id))
                average_FPS += FPS
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

            # # show image
            # cv2.imshow('t', fm)
            # wk = 1
            # if cv2.waitKey(wk) == 27:
            #     break
            # UI_window.setup_control(fm)
            frame_id += 1
            last_frame_time = entry_time

# def run(dict_UI, dict_frame, dict_BC, dict_OT, dict_MOT):
def run(dict_frame, dict_BC, dict_OT, dict_MOT):
# def run(dict_frame, dict_MOT, dict_traj_current, dict_traj_future, dict_BC, dict_OT):
    # from system_main import UI_window
    # while dict_UI['start'] == False:
    #     continue
    t1 = threading.Thread(target=rd_func, args=(dict_frame, dict_BC, dict_OT, dict_MOT,))
    t1.start()
    # UI_window.show()
    # sys.exit(app.exec_())