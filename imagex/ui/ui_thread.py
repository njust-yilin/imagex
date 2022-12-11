from loguru import logger
from PySide6.QtCore import QThread, Signal
from threading import Event
from typing import List
import time

from imagex.settings import configs
from imagex.api.rpc.ui import ui_pb2
from core.communication import ImageSharedMemory


class UIThread(QThread):
    exit_ui = Signal()
    on_image_received = Signal(object)

    def __init__(self):
        super().__init__()
        self.stop_event = Event()
        self.busy = False
        self.start()

    def run(self):
        logger.info('UIThread starting...')
        # TODO: create camera's ImageSharedMemory
        # self.ui_queue = PipeConsumer(UIQueueElement, configs.UI_QUEUE_NAME)
        self.images_comsumer = ImageSharedMemory(configs.IMAGEX_IMAGE_IMAGE_NAME, (2048, 2448, 3), 20)
        self.masks_comsumer = ImageSharedMemory(configs.IMAGEX_MASK_IMAGE_NAME, (2048, 2448, 3), 20)
        logger.info('UIThread Started')
        while not self.stop_event.is_set():
            time.sleep(1)
        logger.info('UIThread quited')

    def image_ready(self, request:ui_pb2.ImageUpdateRequest):
        self.on_image_received.emit(request)

    def exit(self):
        self.stop_event.set()
        self.wait()
        self.exit_ui.emit()
        # del self.images_comsumer
        # del self.ui_queue