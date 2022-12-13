from PySide6.QtWidgets import QStackedWidget, QWidget, QHBoxLayout
from threading import RLock
from loguru import logger

from imagex.ui.components.labels import ImageLabel
from imagex.ui.ui_thread import UIThread
from imagex.api.rpc.imagex import imagex_pb2

class ImageWidget(QStackedWidget):
    def __init__(self, parent, ui_thread:UIThread) -> None:
        super().__init__(parent)
        self.ui_thread = ui_thread
        self.ui_thread.on_image_received.connect(self.update_image)
        self.lock = RLock()
        self.setup_ui()

    def setup_ui(self):
        # self.image_view_widget = QWidget(self)
        # self.image_view_layout = QHBoxLayout()
        # self.image_view_widget.setLayout(self.image_view_layout)
        self.image_label = ImageLabel()
        self.addWidget(self.image_label)

    def update_image(self, request:imagex_pb2.UpdateImageRequest):
        with self.lock:
            logger.info(f"updating image {request.image_index}")
            image = self.ui_thread.images_client.get(request.image_index)
            self.image_label.set_image(image, request.ok)