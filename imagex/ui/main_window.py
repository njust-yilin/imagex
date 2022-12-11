from PySide6.QtWidgets import QMainWindow, QMessageBox, QStackedWidget
from PySide6 import QtGui
from loguru import logger

from imagex.ui.pages.home_page import HomePage
from imagex.ui.ui_thread import UIThread
from imagex.ui import Pages
from core.utils import get_screen_size

class MainWindow(QMainWindow):
    def __init__(self, ui_thread:UIThread):
        super().__init__()
        self.ui_thread = ui_thread
        self.setup_ui()

    def setup_ui(self):
        # setup size
        self.setFixedSize(get_screen_size())

        # setup pages
        self.pages = QStackedWidget(self)
        self.pages.setAutoFillBackground(True)
        self.pages.addWidget(HomePage(self, self.ui_thread))
        
        self.setCentralWidget(self.pages)
        # show full screen
        self.showFullScreen()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if(self.ui_thread.busy):
            res = QMessageBox.question(self, "Quit?", "Busy, confirm to quit")
            if(res == QMessageBox.No):
                event.ignore()
                return
        logger.info("Main window closing...")
        self.ui_thread.exit()
        return super().closeEvent(event)