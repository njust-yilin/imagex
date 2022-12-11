from PySide6 import QtGui


def get_screen_size():
    screen = QtGui.QGuiApplication.primaryScreen()
    return screen.size()