from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel

from deepx.cvlibs import NetworkType


class NetworkItem(QWidget):
    def __init__(
        self, parent=None, network:str='', 
        network_type:NetworkType=NetworkType.Segment):
        super().__init__(parent)

        self.network_type = network_type
        self.network = network

        layout = QHBoxLayout()
        self.name_label = QLabel(text=self.network)
        self.icon_label = QLabel(text=self.network_type.name)
        layout.addWidget(self.name_label)
        layout.addWidget(self.icon_label)

        self.setLayout(layout)