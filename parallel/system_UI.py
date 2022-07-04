import sys
import cv2
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog
from PyQt5.QtWidgets import QGridLayout, QLabel, QPushButton, QWidget, QCheckBox, QLineEdit
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import pyqtSlot
from PyQt5 import QtWidgets, QtCore
import numpy as np
import os
import threading
import time
import torch.multiprocessing as torch_mp

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        width, height = 1100, 550
        MainWindow.resize(width, height)
        
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.move(100, 0)
        # label
        self.label_vid = QtWidgets.QLabel(self.centralwidget)
        self.label_vid.setGeometry(QtCore.QRect(30, 30, 1080, 720))
        self.label_vid.setObjectName("label_vid")
        
        self.label_id = QtWidgets.QLabel(self.centralwidget)
        self.label_id.setObjectName("ID selector")
        self.label_id.move(700, 155)
        self.label_id.setFont(QFont('Arial', 14))
        self.label_opened = QtWidgets.QLabel(self.centralwidget)
        self.label_opened.setObjectName("label_opened_file")
        self.label_opened.move(820, 435)
        self.label_opened.setFont(QFont('Arial', 14))
        self.label_suggestion = QtWidgets.QLabel(self.centralwidget)
        self.label_suggestion.setObjectName("label_suggest")
        self.label_suggestion.move(700, 370)
        self.label_suggestion.setFont(QFont('Arial', 16))
        # self.label_fps = QtWidgets.QLabel(self.centralwidget)
        # self.label_fps.setObjectName("label_fps")
        # self.label_fps.move(820, 54)

        MainWindow.setCentralWidget(self.centralwidget)
        # barbarbar
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, width, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        # check box 
        self.checkbox_disp = QCheckBox('Display', self.centralwidget)
        self.checkbox_disp.move(700, 50)
        self.checkbox_disp.setChecked(False)
        self.checkbox_disp.setFont(QFont('Arial', 14))
        self.checkbox_MOT = QCheckBox('MOT', self.centralwidget)
        self.checkbox_MOT.move(700, 100)
        self.checkbox_MOT.setChecked(False)
        self.checkbox_MOT.setEnabled(False)
        self.checkbox_MOT.setFont(QFont('Arial', 14))
        self.checkbox_TP1 = QCheckBox('History TP', self.centralwidget)
        self.checkbox_TP1.move(810, 100)
        self.checkbox_TP1.setChecked(False)
        self.checkbox_TP1.setEnabled(False)
        self.checkbox_TP1.setFont(QFont('Arial', 14))
        self.checkbox_TP2 = QCheckBox('Future TP', self.centralwidget)
        self.checkbox_TP2.move(970, 100)
        self.checkbox_TP2.setChecked(False)
        self.checkbox_TP2.setEnabled(False)
        self.checkbox_TP2.setFont(QFont('Arial', 14))
        self.checkbox_TP = QCheckBox('active TP module', self.centralwidget)
        self.checkbox_TP.move(700, 200)
        self.checkbox_TP.setChecked(True)
        self.checkbox_TP.setFont(QFont('Arial', 14))
        self.checkbox_OT = QCheckBox('active OT module', self.centralwidget)
        self.checkbox_OT.move(700, 250)
        self.checkbox_OT.setChecked(True)
        self.checkbox_OT.setFont(QFont('Arial', 14))
        # button
        self.btn_start = QtWidgets.QPushButton(self.centralwidget)
        self.btn_start.setObjectName("start_btn")
        self.btn_start.move(700, 480)
        self.btn_start.setFont(QFont('Arial', 14))

        self.btn_stop = QtWidgets.QPushButton(self.centralwidget)
        self.btn_stop.setObjectName("stop_btn")
        self.btn_stop.move(800, 480)
        self.btn_stop.setFont(QFont('Arial', 14))

        self.file_button = QtWidgets.QPushButton(self.centralwidget)
        self.file_button.setObjectName("Open file")
        self.file_button.move(700, 430)
        self.file_button.setFont(QFont('Arial', 14))
        self.btn_save = QtWidgets.QPushButton(self.centralwidget)
        self.btn_save.setObjectName("save_btn")
        self.btn_save.move(900, 480)
        self.btn_save.setFont(QFont('Arial', 14))
        self.btn_quit = QtWidgets.QPushButton(self.centralwidget)
        self.btn_quit.setObjectName("quit_btn")
        self.btn_quit.move(1000, 480)
        self.btn_quit.setFont(QFont('Arial', 14))
        self.btn_track = QtWidgets.QPushButton(self.centralwidget)
        self.btn_track.setObjectName("track_btn")
        self.btn_track.move(900, 149)
        self.btn_track.setFont(QFont('Arial', 14))
        self.btn_track_clear = QtWidgets.QPushButton(self.centralwidget)
        self.btn_track_clear.setObjectName("track_clear_btn")
        self.btn_track_clear.move(990, 149)
        self.btn_track_clear.setFont(QFont('Arial', 14))
        # textbox
        self.textbox_MOT_ID = QLineEdit(self.centralwidget)
        self.textbox_MOT_ID.move(730, 155)
        self.textbox_MOT_ID.setEnabled(False)


        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_vid.setText(_translate("MainWindow", "TextLabel"))
        self.label_opened.setText(_translate("MainWindow", "./"))
        self.label_id.setText(_translate("MainWindow", "ID: "))
        self.label_suggestion.setText(_translate("MainWindow", "Suggestion: "))
        # self.label_fps.setText(_translate("MainWindow", "FPS:"))

        self.btn_start.setText(_translate("MainWindow", "Start"))
        self.btn_stop.setText(_translate("MainWindow", "Stop"))
        self.file_button.setText(_translate("MainWindow", "Open File"))
        self.btn_save.setText(_translate("MainWindow", "Save"))
        self.btn_quit.setText(_translate("MainWindow", "Quit"))
        self.btn_track.setText(_translate("MainWindow", "Track"))
        self.btn_track_clear.setText(_translate("MainWindow", "Clear"))
        
# ==========================================================================================
class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.manager = torch_mp.Manager()
        self.config = dict()
        
        self.init_mem()
        self.black_screen = np.zeros((480, 640, 3))
        self.set_img(self.black_screen)
        self.setup_control()
        
    def init_config(self):
        self.config['TP'] = self.ui.checkbox_TP.isChecked()
        self.config['OT'] = self.ui.checkbox_OT.isChecked()
        self.config['Exit'] = False # threading.Event()
        self.config['MOT'] = self.ui.checkbox_MOT.isChecked() # threading.Event()
        self.config['HTP'] = self.ui.checkbox_TP1.isChecked()
        self.config['FTP'] = self.ui.checkbox_TP2.isChecked() and self.ui.checkbox_TP.isChecked()
        self.config['ID'] = 0
        self.config['FPS'] = 0

    def init_mem(self):
        # share memory config =========================================
        self.dict_frame = self.manager.dict() # save frame
        self.dict_objdet = self.manager.dict() # save objdet result
        self.dict_MOT = self.manager.dict() # save MOT result
        self.dict_traj_id_dict = self.manager.dict() # save traj by format {id : traj}
        self.dict_traj_future = self.manager.dict()
        self.dict_BC = self.manager.dict()
        self.dict_OT = self.manager.dict()
        self.lock = self.manager.Lock()
        
        # share memory config =========================================
    def init_processes(self):
        import torch.multiprocessing as torch_mp
        from Processor_1 import run as P1_run
        from Processor_2 import run as OT_run
        from Processor_3 import run as Input_reader
        from Processor_4 import run as Output_reader
        # create subprocess
        self.p_list = [[]] * 3
        self.p_list[0] = torch_mp.Process(target=OT_run, 
        args=(self.dict_frame, self.dict_objdet, self.dict_OT,self.lock,))

        self.p_list[1] = torch_mp.Process(target=P1_run,
        args=(self.dict_frame, self.dict_objdet,\
             self.dict_traj_future, self.dict_BC, self.dict_MOT, self.config, self.lock,))
        
        self.p_list[2] = torch_mp.Process(target=Input_reader,\
             args=(self.config['video_path'], self.dict_frame, self.lock,))
        
        self.t1 = threading.Thread(target=Output_reader, 
        args=(self.dict_frame, self.dict_MOT, self.dict_traj_future,\
             self.dict_BC, self.dict_OT, self.lock, self.config, self.set_img, self.set_fps,))
        # start each subprocess
    def set_fps(self, fps):
        self.ui.label_fps.setText("FPS: {}".format(fps))
        self.ui.label_fps.adjustSize()

    def set_img(self, fm):
        self.fm = cv2.resize(fm, (640, 480))
        height, width = self.fm.shape[:2]
        bytesPerline = 3 * width
        self.qimg = QImage(self.fm, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.ui.label_vid.setPixmap(QPixmap.fromImage(self.qimg))
        self.ui.label_vid.adjustSize()

    def utilities_set_all(self, flag):
        # close three check box
        self.ui.checkbox_OT.setEnabled(flag)
        self.ui.checkbox_TP.setEnabled(flag)
        self.ui.file_button.setEnabled(flag)

    def start_func(self):
        video_ext = ['MOV', 'mov', 'avi', 'AVI', 'mp4', 'MP4']
        # check open file type is above type.
        if 'video_path' not in self.config or \
        self.config['video_path'].split('.')[-1] not in video_ext:
            self.ui.label_opened.setStyleSheet('color:red')
            self.ui.label_opened.setText("Please Select a File.")
            self.ui.label_opened.adjustSize()
            return 

        self.init_config()
        self.utilities_set_all(False)
        self.init_mem()
        self.init_processes()
        for i in range(3):
            # don't open OT module
            if not self.config['OT'] and i == 0:
                continue
            self.p_list[i].start()
            if i == 0:
                time.sleep(8)
        self.t1.start()
    
    def stop_func(self):
        try:
            self.config['Exit'] = True
            self.t1.join()
            # open all checkbox
            self.utilities_set_all(True)
            # don't open OT module
            for i in range(len(self.p_list)):
                if not self.config['OT'] and i == 0:
                    continue
                self.p_list[i].terminate()
            self.set_img(self.black_screen)

        except Exception as e:
            print(type(e).__name__, e)
        
    def setup_control(self):
        self.ui.btn_start.clicked.connect(self.start_func)
        self.ui.btn_stop.clicked.connect(self.stop_func)
        self.ui.btn_save.clicked.connect(self.save_display_result)
        self.ui.file_button.clicked.connect(self.open_file) 
        self.ui.btn_track.clicked.connect(self.ID_select)
        self.ui.btn_track_clear.clicked.connect(self.clear_ID)
        self.ui.checkbox_TP.stateChanged.connect(self.active_TP)
        self.ui.checkbox_OT.stateChanged.connect(self.active_OT)
        self.ui.checkbox_MOT.stateChanged.connect(self.MOT_cb)
        self.ui.checkbox_TP1.stateChanged.connect(self.TP1_cb)
        self.ui.checkbox_TP2.stateChanged.connect(self.TP2_cb)
        self.ui.checkbox_disp.stateChanged.connect(self.disp_config)
        self.ui.btn_quit.clicked.connect(self.close_App)
        
    def open_file(self):
        filename, filetype = QFileDialog.getOpenFileName(self, "Open file", "../../../Dataset/CEO/20201116/front/")
        if filename != '':
            self.config['video_path'] = filename
            self.ui.label_opened.setText(filename.split('/')[-1])
            self.ui.label_opened.setStyleSheet('color:black')
            self.ui.label_opened.adjustSize()
        print(filename, filetype)
    
    def MOT_cb(self):
        self.config['MOT'] = self.ui.checkbox_MOT.isChecked()
    def TP1_cb(self):
        self.config['HTP'] = self.ui.checkbox_TP1.isChecked()
    def TP2_cb(self):
        self.config['FTP'] = self.ui.checkbox_TP2.isChecked() and self.ui.checkbox_TP.isChecked()

    def ID_select(self):
        ID = self.ui.textbox_MOT_ID.text()
        if ID == '':
            self.config['ID'] = 0
        else:
            self.config['ID'] = int(ID)
    def clear_ID(self):
        self.config['ID'] = 0
        self.ui.textbox_MOT_ID.clear()
        
    def active_TP(self):
        self.config['TP'] = self.ui.checkbox_TP.isChecked()
        
    def active_OT(self):
        self.config['OT'] = self.ui.checkbox_OT.isChecked()

    def disp_config(self, flag=None):
        disp_state = self.ui.checkbox_disp.isChecked()
        if flag is not None:
            disp_state = flag
        self.ui.checkbox_MOT.setEnabled(disp_state)
        self.ui.checkbox_TP1.setEnabled(disp_state)
        self.ui.checkbox_TP2.setEnabled(disp_state and self.ui.checkbox_TP.isChecked())
        self.ui.textbox_MOT_ID.setEnabled(disp_state)
        if disp_state == False:
            self.config['MOT'] = False
            self.config['HTP'] = False
            self.config['FTP'] = False
        else:
            self.config['MOT'] = self.ui.checkbox_MOT.isChecked()
            self.config['HTP'] = self.ui.checkbox_TP1.isChecked()
            self.config['FTP'] = self.ui.checkbox_TP2.isChecked() and self.ui.checkbox_TP.isChecked()

    def save_display_result(self):
        print("Save display Result, TODO")

    def close_App(self):
        self.stop_func()
        self.close()

