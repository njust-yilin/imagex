import numpy as np
import time
from threading import Thread
from threading import Event
from loguru import logger
import cv2
from random import randint

from lightx.device.camera.abstract_camera import AbstractCamera
from core.utils import get_timestamp_ms


class FakeCamera(AbstractCamera):
    def __init__(self, serial_number, image_width, image_height) -> None:
        super().__init__(serial_number, image_width, image_height)
        self.trigger_event = Event()

    @staticmethod
    def get_cameras():
        return ['Fake0001', 'Fake0002', 'Fake000']
    
    def initialize(self):
        pass

    def start_grab(self):
        logger.info(f"camera[{self.serial_number}] start grab")
        self.stop_event.clear()
        Thread(target=self.async_process_images).start()
        Thread(target=self.fake_service).start()

    def __del__(self):
        self.stop_grab()


    def stop_grab(self):
        logger.info(f"camera[{self.serial_number}] stop grabbing")
        self.stop_event.set()

    def trigger(self):
        with self.lock:
            self.trigger_event.set()
            logger.info(f"camera[{self.serial_number}] trigger")

    def image_convert(self, image:np.ndarray)->np.ndarray:
        return image

    def fake_service(self):
        images = np.zeros((10, self.height, self.width, 3), dtype=np.uint8)
        for i in range(10):
            images[i] = cv2.putText(images[i], f"Test-{i}", (200, 1000), cv2.FONT_HERSHEY_COMPLEX, 12, (0, 0, 255), 2)

        while not self.stop_event.is_set():
            if self.trigger_event.is_set():
                with self.lock:
                    self.trigger_event.clear()
                    ts = get_timestamp_ms()
                    self.image_queue.put((ts, images[randint(0, 9)]))
                    logger.info(f"camera[{self.serial_number}] generate one image")
            else:
                time.sleep(0.01)
        logger.info(f"camera[{self.serial_number}] Stopped")
