from PySide6.QtWidgets import QWidget

from imagex.ui.ui_thread import UIThread
from core.utils import get_screen_size
from imagex.ui.components.image_widget import ImageWidget


class HomePage(QWidget):
    def __init__(self, parent, ui_thread:UIThread):
        super().__init__(parent)
        self.ui_thread = ui_thread
        size = get_screen_size()
        w, h = size.width(), size.height()
        self.resize(get_screen_size())

        self.image_widget = ImageWidget(self, self.ui_thread)
        self.image_widget.setGeometry(0, 0, w, h)