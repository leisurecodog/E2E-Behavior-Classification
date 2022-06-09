from os import system
import torch.multiprocessing as mp
# mp.set_start_method('spawn')
import cv2
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
            stop_flag = False
            if sys_args.show:
                stop_flag = System.show(frame)
            if stop_flag:
                break
            frame_id += 1
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
    # import torch.multiprocessing as mp
    # mp.set_start_method('spawn')
    import system_parser
    import system_class
    # import threading
    # import numpy as np
    # import torch
    # from OT_module.yolact_edge_project.utils.augmentations import FastBaseTransform
    # t_list = []
    # from OT_module.yolact_edge_project.eval import load_yolact_edge, prep_display
    # frame = np.zeros((640,480,3))
    # print(frame.shape)
    # frame_tensor = torch.from_numpy(frame).cuda().float()
    # batch = FastBaseTransform()(frame_tensor.unsqueeze(0))
    # moving_statistics = {"conf_hist": []}
    # extras = {"backbone": "full", "interrupt": False, "keep_statistics": False,"moving_statistics": moving_statistics}

    # yolact = load_yolact_edge()
    # yolact.detect.use_fast_nms = True
    # for i in range(10):
        # p = mp.Process(target=yolact, args=(batch, extras,))
        # p.start()
        # p.join()
    run()
    # t_ui = threading.Thread(target=window)
    # t_list.append(t_ui)
    
    # t_sys = threading.Thread(target=run)
    # t_list.append(t_sys)

    # for t in t_list:
        # t.start()
    # for t in t_list:
        # t.join()
