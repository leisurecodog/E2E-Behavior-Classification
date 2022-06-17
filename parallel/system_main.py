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
    from MOT_module.MOT_processor import run as MOT_run
    from TP_module.TP_processor import run as TP_run
    global g_frame
    sys_args = system_parser.get_parser()
    cap = cv2.VideoCapture(sys_args.video_path if sys_args.demo == "video" else sys_args.camid)
    frame_id = 0
    manager = Manager()
    dict_frame = manager.dict()
    dict_objdet = manager.dict()
    dict_MOT = manager.dict()
    list_traj_current = manager.list()
    dict_traj_future = manager.dict()
    p_list = [[]] * 20
    p_list[0] = Process(target=MOT_run, args=(dict_frame, dict_objdet, dict_MOT,))
    p_list[0].start()
    p_list[1] = Process(target=TP_run, args=(dict_MOT, list_traj_current, dict_traj_future,))
    p_list[1].start()
    while True:
      # ret_val: True -> Read image, False -> No image
      # frame: image frame.
        ret_val, frame = cap.read()
        # sys.reset = False
        
      # start working when have image.
        if ret_val:
            g_frame = frame.copy()
            t_time_1 = time.time()            
            if sys_args.resize:
                frame = cv2.resize(frame, (sys_args.size))
            # bounding box and ID infomation
            dict_frame[frame_id] = frame
            # if sys.BC.is_satisfacation(sys.TP.ID_counter):
            #     sys.reset = True
            #     sys.BC.run(sys.TP.traj, sys.TP.future_trajs)
            # sys.OT.run(sys.MOT.objdet, frame)
            # print(sys.OT.OTS.msg)
            # # frame = System.OT_run(frame) # for debug using.
            # stop_flag = False
            # if sys_args.show:
            #     stop_flag = sys.show(frame, t_time_1)
            # if stop_flag:
            #     cap.release()
            #     cv2.destroyAllWindows()
            #     break
            frame_id += 1
            # if sys.reset:
            #     sys.TP.traj_reset()
        else:
            print("video is end.")
            break
    # print("MOT average time:", sys.MOT.exe_time/sys.MOT.counter)
    # print("TP average time:", sys.TP.exe_time/sys.TP.counter)
    # print("BC average time:", sys.BC.exe_time/sys.BC.counter)
    # print("OT average time:", sys.OT.exe_time/sys.OT.counter)

def window():
    import sys
    app = QApplication(sys.argv)
    dialog = MyDialog()
    dialog.show()
    dialog.set_img()
    sys.exit(app.exec_())

def show(self, frame, t_total=None):
        bbox = self.MOT.result
        MOT_show_flag = False
        future_traj_flag = True if self.TP.future_trajs is not None else False
        BC_result_flag = True if self.BC.result is not None else False

        # Draw bounding box & agent ID and Behavior Classification Result
        # draw_color_idx: 0 mean normal, 1 mean aggressive
        if MOT_show_flag:
            for id in bbox.keys():
                draw_color_idx = 0
                x1, y1, offset_x, offset_y = bbox[id]
                x2, y2 = x1 + offset_x, y1 + offset_y
                if BC_result_flag and id in self.BC.result.keys():
                    bcr = self.BC.result[id]
                    if bcr == -1:
                        draw_color_idx = 1
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), self.BC.bbox_color[draw_color_idx], 2)
                cv2.putText(frame, str(id), (int(x1), int(y1)), cv2.FONT_HERSHEY_PLAIN,\
                        1, (255, 255, 0), thickness=1)

        # Draw trajectory using point
        traj_show_flag = True
        try:
            # Draw past traj
            if traj_show_flag:
                for f in self.TP.traj:
                    for k, v in f.items():
                        cv2.circle(frame, (int(v[0]), int(v[1])), 3, (0,0,255), -1)
            # Draw future traj
            if self.TP.future_trajs is not None and traj_show_flag:
                for k, v in self.TP.future_trajs.items():
                    for x, y in v:
                        cv2.circle(frame, (int(x), int(y)), 3, (255,0,0), -1)
            if t_total is not None:
                cv2.putText(frame, "FPS: "+str(1.0/(time.time()-t_total)), (0, 20),cv2.FONT_HERSHEY_PLAIN,\
                        1, (255,255,255), thickness=1)
            # wk: if future_traj is drawn, then waitkey set 0 for better visualization.
            wk = 0 if future_traj_flag else 1
            self.writer.write(frame)
            cv2.imshow('t', frame)
            wk = 1
            if cv2.waitKey(wk) == 27: # whether is pressed ESC key.
                print("ESC pressed.")
                return True
        except Exception as e:
            print("Exception is happened: {}".format(e))

if __name__ == '__main__':
    import system_parser
    import system_class
    from torch.multiprocessing import Process, Manager
    import torch.multiprocessing as mp
    mp.set_start_method('spawn')
    run()
    # window()
