from os import system
import cv2
from system_UI import MyDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QGridLayout, QLabel, QPushButton, QWidget
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
import time
import numpy as np
g_frame = np.zeros((1080, 720, 3))

def run():
    global g_frame
    sys_args = system_parser.get_parser()
    sys = system_class.DrivingBehaviorSystem(sys_args)
    # sys = system_class_parallel.DrivingBehaviorSystem(sys_args)
    cap = cv2.VideoCapture(sys_args.video_path if sys_args.demo == "video" else sys_args.camid)
    frame_id = 0
    interval = 5
    while True:
      # ret_val: True -> Read image, False -> No image
      # frame: image frame.
        ret_val, frame = cap.read()
        sys.reset = False
      # start working when have image.
        if ret_val:
            g_frame = frame.copy()
            t_time_1 = time.time()            
            if sys_args.resize:
                frame = cv2.resize(frame, (sys_args.size))
            # bounding box and ID infomation
            sys.MOT.run(frame)
            
            sys.TP.update_traj(sys.MOT.result)
            
            if sys_args.future and frame_id % interval == 0:
                sys.TP.run()
            if sys.BC.is_satisfacation(sys.TP.ID_counter):
                sys.reset = True
                sys.BC.run(sys.TP.traj, sys.TP.future_trajs)
            sys.OT.run(sys.MOT.objdet, frame)
            print(sys.OT.OTS.msg)
            # frame = System.OT_run(frame) # for debug using.
            stop_flag = False
            if sys_args.show:
                stop_flag = sys.show(frame, t_time_1)
            if stop_flag:
                cap.release()
                cv2.destroyAllWindows()
                break
            frame_id += 1
            if sys.reset:
                sys.TP.traj_reset()
        else:
            print("video is end.")
            break
    print("MOT average time:", sys.MOT.exe_time/sys.MOT.counter)
    print("TP average time:", sys.TP.exe_time/sys.TP.counter)
    print("BC average time:", sys.BC.exe_time/sys.BC.counter)
    print("OT average time:", sys.OT.exe_time/sys.OT.counter)
    print('')
    print("YOLO average time:", sys.MOT.yolo_time/sys.MOT.yolo_counter)
    print("Tracker average time:", sys.MOT.tracker_time/sys.MOT.tracker_counter)
    print("main RQI average time:", sys.BC.main_time/sys.BC.main_counter)
    print("PInet average time:", sys.OT.pinet_time/sys.OT.pinet_counter)
    print("Yolact average time:", sys.OT.yolact_time/sys.OT.yolact_counter)
    print("DDPG average time:", sys.TP.single_traj_time/sys.TP.traj_pred_counter)

def window():
    import sys
    app = QApplication(sys.argv)
    dialog = MyDialog()
    dialog.show()
    dialog.set_img()
    sys.exit(app.exec_())

if __name__ == '__main__':
    import system_parser
    import system_class
    import system_class_parallel
    run()
    # window()