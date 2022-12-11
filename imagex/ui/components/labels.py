from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QFrame
import numpy as np


class ImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__()
        self.setMinimumSize(5, 5)
        
    def set_image(self, image:np.ndarray, result:bool):
        h, w = image.shape[0], image.shape[1]
        qimage = QImage(image, w, h, QImage.Format.Format_RGB888)
        qpixmap = QPixmap.fromImage(qimage)
        self.setFrameShape(QFrame.Shape.Box)
        self.setFrameShadow(QFrame.Shadow.Raised)
        if result:
            self.setStyleSheet('border-width: 2px; border-style: solid; border-color: rgb(0,255,0)')
        else:
            self.setStyleSheet('border-width: 2px; border-style: solid; border-color: rgb(255,0,0)')

        self.setPixmap(qpixmap)