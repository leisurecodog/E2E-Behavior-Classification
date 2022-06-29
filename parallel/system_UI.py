import sys
import cv2
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QGridLayout, QLabel, QPushButton, QWidget, QCheckBox
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
from PyQt5 import QtWidgets, QtCore
import numpy as np

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1280, 550)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.move(100, 0)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(30, 30, 1080, 720))
        self.label.setObjectName("label")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1280, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.checkbox = QCheckBox('Activate TP module', self.centralwidget)
        self.checkbox.move(700, 200)
        self.btn_start = QtWidgets.QPushButton(self.centralwidget)
        self.btn_start.setObjectName("start_btn")
        self.btn_start.move(700, 480)
        self.btn_hello = QtWidgets.QPushButton(self.centralwidget)
        self.btn_hello.setObjectName("stop_btn")
        self.btn_hello.move(800, 480)
        
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "TextLabel"))
        self.btn_start.setText(_translate("MainWindow", "Start"))
        self.btn_hello.setText(_translate("MainWindow", "Stop"))

class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        ze = np.zeros((480, 640, 3))
        self.set_img(ze)
        self.setup_control()

    def set_component(self, dic, t1, stop_event):
        self.share_dict = dic
        self.t1 = t1
        self.stop_event = stop_event
        
    def set_img(self, fm):
        if fm is not None:
            self.fm = cv2.resize(fm, (640, 480))
            self.change_img()

    def start_func(self):
        self.share_dict['start'] = True
        self.t1.start()
    
    def stop_func(self):
        self.stop_event.set()

    def setup_control(self):
        self.ui.btn_start.clicked.connect(self.start_func)
        self.ui.btn_hello.clicked.connect(self.stop_func)

    def display_img(self):
        self.img = cv2.imread(self.img_path)
        height, width, channel = self.img.shape
        bytesPerline = 3 * width
        self.qimg = QImage(self.img, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.ui.label.setPixmap(QPixmap.fromImage(self.qimg))
        self.ui.label.adjustSize()

    def change_img(self):
        height, width, channel = self.fm.shape
        bytesPerline = 3 * width
        self.qimg = QImage(self.fm, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.ui.label.setPixmap(QPixmap.fromImage(self.qimg))
        self.ui.label.adjustSize()
