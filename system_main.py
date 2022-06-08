from os import system
import cv2
import system_parser
import system_class
from system_UI import MyDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QGridLayout, QLabel, QPushButton, QWidget
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot

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
            futures = None
            if sys_args.resize:
                frame = cv2.resize(frame,(sys_args.size))
            # bounding box and ID infomation
            System.MOT_run(frame, frame_id, format=sys_args.format_str)

            System.update_traj()
            if sys_args.future:
                System.get_future_traj()

            System.BC_run()

            stop_flag = System.OT_run(frame)
            if stop_flag:
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
def window():
    import sys
    app = QApplication(sys.argv)
    dialog = MyDialog()
    dialog.show()
    dialog.set_img(g_frame)
    sys.exit(app.exec_())

if __name__ == '__main__':
    import threading
    t_list = []
    # t_ui = threading.Thread(target=window)
    # t_list.append(t_ui)
    
    t_sys = threading.Thread(target=run)
    t_list.append(t_sys)

    for t in t_list:
        t.start()
    for t in t_list:
        t.join()
