from PySide6.QtWidgets import QMainWindow, QStackedWidget
from PySide6.QtCore import Qt

from deepx.ui.widgets import Pages, HomeWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._stack_widget = QStackedWidget(self)
        self.setMinimumSize(900, 600)

        # background
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        # add widgets
        self._stack_widget.addWidget(HomeWidget())

        self.setCentralWidget(self._stack_widget)
        self.showMaximized()
    
    def change_page(self, page:Pages):
        self._stack_widget.setCurrentIndex(page.value)

if __name__ == '__main__':
    import sys
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    app.exec()