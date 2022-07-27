import sys
import cv2
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot, QThread, pyqtSignal, QTimer

class ProcessWorker(QObject):
    imageChanged = pyqtSignal(QImage)

    def doWork(self):
        for f in self.image_list:
            img = cv2.imread(f)
            img = QImage(img, False)
            self.imageChanged.emit(img)

class Widget(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        lay = QVBoxLayout(self)
        gv = QGraphicsView()
        lay.addWidget(gv)
        scene = QGraphicsScene(self)
        gv.setScene(scene)
        self.pixmap_item = QGraphicsPixmapItem()
        scene.addItem(self.pixmap_item)
        self.timer = QTimer(self)
        self.timer.setSingleShot(False)
        self.timer.timeout.connect(self.worker)
    
    def play(self):
        self.timer.start(1000)

    @pyqtSlot(QImage)
    def setImage(self, image):
        pixmap = QPixmap.fromImage(image)
        self.pixmap_item.setPixmap(pixmap)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Widget()
    w.show()
    sys.exit(app.exec_())