from os import system
import cv2
import system_parser
import system_class
# MOT: Mulitple Object Tracking
# TP : Trajectory Prediction
# BC : Bahavior Classification
# OT : Overtaking Assistant
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot

def window():
   app = QApplication(sys.argv)
   widget = QWidget()

   textLabel = QLabel(widget)
   textLabel.setText("Hello World!")
   textLabel.move(110,85)

   widget.setGeometry(50,50,320,200)
   widget.setWindowTitle("PyQt5 Example")
   widget.show()
   sys.exit(app.exec_())
   
if __name__ == '__main__':
   #  window()
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
            futures = None
            if sys_args.resize:
                frame = cv2.resize(frame,(sys_args.size))
            # bounding box and ID infomation
            # ========== run MOT module ==========
            System.MOT_run(frame, frame_id, format=sys_args.format_str)

            # ========== run TP module ==========
            System.update_traj()
            if sys_args.future:
                System.get_future_traj()
            # ========== run BC module ==========
            System.BC_run()

            # ========== run OT module ==========
            if System.OT_run(frame):
                break
            flag = False
            if sys_args.show:
                flag = System.show(frame)
            if flag:
                break
            frame_id += 1
            if System.traj_reset_flag:
                System.traj_reset()
                System.traj_reset_flag = False
        else:
            print("video is end.")
            break
