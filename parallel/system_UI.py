import sys
import cv2
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QGridLayout, QLabel, QPushButton, QWidget
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot

def window():
   app = QApplication(sys.argv)
   # widget = QWidget()

   # textLabel = QLabel(widget)
   # textLabel.setText("Hello World!")
   # textLabel.move(110,85)

   # widget.setGeometry(50,50,320,200)
   # widget.setWindowTitle("PyQt5 Example")
   # widget.show()
   dialog = MyDialog()
   dialog.show()
   sys.exit(app.exec_())

class MyDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def set_img(self, img):
        self.img = img

    def initUI(self):
        self.resize(400, 300)
        self.label = QLabel()
        self.btnOpen = QPushButton('Open Image', self)
        self.btnProcess = QPushButton('Blur Image', self)
        self.btnSave = QPushButton('Save Image', self)
        self.btnSave.setEnabled(False)

        layout = QGridLayout(self)
        layout.addWidget(self.label, 0, 0, 4, 4)
        layout.addWidget(self.btnOpen, 4, 0, 1, 1)
        layout.addWidget(self.btnProcess, 4, 1, 1, 1)
        layout.addWidget(self.btnSave, 4, 2, 1, 1)

        self.btnOpen.clicked.connect(self.openSlot)
        self.btnProcess.clicked.connect(self.processSlot)
        self.btnSave.clicked.connect(self.saveSlot)

    def openSlot(self):
        # filename, _ = QFileDialog.getOpenFileName(self, 'Open Image', 'Image', '*.png *.jpg *.bmp')
        # if filename is '':
        #     return
        # self.img = cv2.imread(filename, -1)
        # if self.img.size == 1:
        #     return
        self.showImage()
        self.btnSave.setEnabled(True)

    def saveSlot(self):
        filename, _ = QFileDialog.getSaveFileName(self, 'Save Image', 'Image', '*.png *.jpg *.bmp')
        if filename is '':
            return
        cv2.imwrite(filename, self.img)

    def processSlot(self):
        self.img = cv2.blur(self.img, (7, 7))
        self.showImage()

    def showImage(self):
        height, width, channel = self.img.shape
        bytesPerline = 3 * width
        self.qImg = QImage(self.img.data, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.label.setPixmap(QPixmap.fromImage(self.qImg))
