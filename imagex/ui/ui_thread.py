from loguru import logger
from PySide6.QtCore import QThread, Signal, QCoreApplication
from threading import Event
from typing import List
import time

from imagex.settings import configs
from imagex.api.rpc import imagex_pb2
from core.communication import ImageSharedMemoryClient


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
        logger.info('UIThread Started')
        while not self.stop_event.is_set():
            logger.info('UIThread routine...')
            time.sleep(1)
        logger.info('UIThread quitted')

    def initialize(self):
        # TODO: create camera's ImageSharedMemory
        logger.info('Initializing UI Thread shared memory')
        self.images_client = ImageSharedMemoryClient(configs.IMAGEX_IMAGE_IMAGE_NAME, (2048, 2448, 3), 20)
        self.masks_client = ImageSharedMemoryClient(configs.IMAGEX_MASK_IMAGE_NAME, (2048, 2448, 3), 20)

    def image_ready(self, request:imagex_pb2.UpdateImageRequest):
        self.on_image_received.emit(request)

    def exit(self):
        self.stop_event.set()
        self.wait()
        logger.info("Emit exit ui signal")
        self.exit_ui.emit()
        QCoreApplication.quit()
        logger.info('Cleanup UI Thread shared memory')
        if self.images_client:
            del self.images_client
        if self.masks_client:
            del self.masks_client