from os import system
import cv2
from system_UI import MyDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QGridLayout, QLabel, QPushButton, QWidget
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
import time
g_frame = None

def run():
    sys_args = system_parser.get_parser()
    sys = system_class.DrivingBehaviorSystem()
    cap = cv2.VideoCapture(sys_args.video_path if sys_args.demo == "video" else sys_args.camid)
    frame_id = 0
    while True:
      # ret_val: True -> Read image, False -> No image
      # frame: image frame.
        ret_val, frame = cap.read()
      # start working when have image.
        if ret_val:
            print("===============================")
            if sys_args.resize:
                frame = cv2.resize(frame, (sys_args.size))
            # bounding box and ID infomation
            sys.MOT.run(frame, frame_id, format=sys_args.format_str)
            # if frame_id % 2 == 0:
            sys.TP.update_traj(sys.MOT.result)
            if sys_args.future:
                sys.TP.run()
            sys.BC.run(sys.TP.ID_counter, sys.TP.traj, sys.TP.future_trajs)
            sys.OT.run(sys.MOT.objdet, frame)
            # frame = System.OT_run(frame) # for debug using.

            if sys_args.show:
                stop_flag = sys.show(frame)
            if stop_flag:
                break
            frame_id += 1
            # print(frame_id)
            if sys.BC.reset_traj_flag:
                sys.TP.traj_reset()
        else:
            print("video is end.")
            break
def window():
    import sys
    app = QApplication(sys.argv)
    dialog = MyDialog()
    dialog.show()
    dialog.set_img(g_frame)
    sys.exit(app.exec_())

if __name__ == '__main__':
    import system_parser
    import system_class
    run()
