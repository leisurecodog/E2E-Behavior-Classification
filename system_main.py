from os import system
import torch.multiprocessing as mp
# mp.set_start_method('spawn')
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
    System = system_class.DrivingBehaviorSystem()
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
            System.MOT_run(frame, frame_id, format=sys_args.format_str)
            # if frame_id % 2 == 0:
            System.update_traj()

            if sys_args.future:
                System.get_future_traj()
            System.BC_run()
            System.OT_run(frame)
            # frame = System.OT_run(frame) # for debug using.

            stop_flag = False
            if sys_args.show:
                stop_flag = System.show(frame)
            if stop_flag:
                break
            frame_id += 1
            # print(frame_id)
            if System.traj_reset_flag:
                System.traj_reset()
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
