
# from multiprocessing.connection import wait
import torch.multiprocessing as torch_mp
import torch
import threading
# from os import system
import cv2
import sys
import time
import numpy as np
from system_util import ID_check
#
from PyQt5 import QtWidgets, QtCore
from system_UI import MainWindow_controller
#

if __name__ == '__main__':
    torch_mp.set_start_method('spawn')
    torch.set_num_threads(1)    
    # run()
    app = QtWidgets.QApplication(sys.argv)
    UI_window = MainWindow_controller()
    UI_window.show()
    sys.exit(app.exec_())