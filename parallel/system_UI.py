import sys
import cv2
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QGridLayout, QLabel, QPushButton, QWidget, QCheckBox
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
from PyQt5 import QtWidgets, QtCore
import numpy as np
import threading
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1280, 550)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.move(100, 0)
        # label
        self.label_vid = QtWidgets.QLabel(self.centralwidget)
        self.label_vid.setGeometry(QtCore.QRect(30, 30, 1080, 720))
        self.label_vid.setObjectName("label_vid")
        self.label_opened = QtWidgets.QLabel(self.centralwidget)
        self.label_opened.setObjectName("label_opened_file")
        self.label_opened.move(800, 435)

        MainWindow.setCentralWidget(self.centralwidget)
        # barbarbar
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1280, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        # check box 
        self.checkbox_TP = QCheckBox('active TP module', self.centralwidget)
        self.checkbox_TP.move(700, 200)
        self.checkbox_OT = QCheckBox('active OT module', self.centralwidget)
        self.checkbox_OT.move(700, 250)
        # button
        self.btn_start = QtWidgets.QPushButton(self.centralwidget)
        self.btn_start.setObjectName("start_btn")
        self.btn_start.move(700, 480)
        self.btn_hello = QtWidgets.QPushButton(self.centralwidget)
        self.btn_hello.setObjectName("stop_btn")
        self.btn_hello.move(800, 480)
        self.file_button = QtWidgets.QPushButton(self.centralwidget)
        self.file_button.setObjectName("Open file")
        self.file_button.move(700, 430)
        self.btn_save = QtWidgets.QPushButton(self.centralwidget)
        self.btn_save.setObjectName("save_btn")
        self.btn_save.move(1000, 480)
        self.btn_quit = QtWidgets.QPushButton(self.centralwidget)
        self.btn_quit.setObjectName("quit_btn")
        self.btn_quit.move(1150, 480)


        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_vid.setText(_translate("MainWindow", "TextLabel"))
        self.label_opened.setText(_translate("MainWindow", "./"))
        self.btn_start.setText(_translate("MainWindow", "Start"))
        self.btn_hello.setText(_translate("MainWindow", "Stop"))
        self.file_button.setText(_translate("MainWindow", "Open File"))
        self.btn_save.setText(_translate("MainWindow", "Save"))
        self.btn_quit.setText(_translate("MainWindow", "Quit"))

class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # self.init_processes() # not finish
        ze = np.zeros((480, 640, 3))
        self.set_img(ze)
        self.setup_control()
        
    def init_processes(self):
        import torch.multiprocessing as torch_mp
        manager = torch_mp.Manager()
        self.dict_frame = manager.dict() # save frame
        self.dict_objdet = manager.dict() # save objdet result
        # self.dict_MOT = manager.dict() # save MOT result
        # self.dict_traj_id_dict = manager.dict() # save traj by format {id : traj}
        # self.dict_traj_future = manager.dict()
        # self.dict_BC = manager.dict()
        self.dict_OT = manager.dict()
        self.dict_UI = manager.dict() # save config
        self.dict_UI['start'] = False
        self.dict_UI['TP'] = False
        self.dict_UI['OT'] = False
        self.dict_UI['stop'] = False
        from Processor_1 import run as P1_run, run2
        from Processor_2 import run as Input_reader
        from Processor_3 import run as Output_reader
        from Processor_4 import p4
        # P1 is MOT-TP-BC Processor, maybe Output & UI will combine in. TODO
        # P2 is OT Processor.
        # P3 is Input Processor, for read image from video.
        self.p1 = threading.Thread(target=P1_run, args=(self.dict_frame, self.dict_objdet))
        self.p2 = torch_mp.Process(target=p4, args=(self.dict_frame, self.dict_objdet, self.dict_OT,))
        self.p3 = torch_mp.Process(target=Input_reader, args=(self.dict_UI, self.dict_frame,))
        
    def set_img(self, fm):
        self.fm = cv2.resize(fm, (640, 480))
        height, width = self.fm.shape[:2]
        bytesPerline = 3 * width
        self.qimg = QImage(self.fm, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.ui.label_vid.setPixmap(QPixmap.fromImage(self.qimg))
        self.ui.label_vid.adjustSize()

    def start_func(self):
        self.share_dict['start'] = True
        self.t1.start()
    
    def stop_func(self):
        self.stop_event.set()

    def setup_control(self):
        self.ui.btn_start.clicked.connect(self.start_func)
        self.ui.btn_hello.clicked.connect(self.stop_func)
        self.ui.btn_save.clicked.connect(self.save_display_result)
        self.ui.file_button.clicked.connect(self.open_file) 
        self.ui.checkbox_TP.stateChanged.connect(self.active_TP)
        self.ui.checkbox_OT.stateChanged.connect(self.active_OT)
        self.ui.btn_quit.clicked.connect(self.close_App)

    def open_file(self):
        filename, filetype = QFileDialog.getOpenFileName(self, "Open file", "../../../Dataset/CEO/20201116")
        if filename != '':
            self.ui.label_opened.setText(filename.split('/')[-1])
            self.ui.label_opened.adjustSize()
        print(filename, filetype)
    
    def active_TP(self):
        # self.ui.dict_UI['TP'] = True
        print(self.ui.checkbox_TP.isChecked())

    def active_OT(self):
        self.ui.dict_UI['OT'] = True

    def save_display_result(self):
        print("Save display Result, TODO")

    def close_App(self):
        self.close()

