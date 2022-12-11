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
        # image = np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)
        while not self.stop_event.is_set():
            if self.trigger_event.is_set():
                with self.lock:
                    self.trigger_event.clear()
                    ts = get_timestamp_ms()
                    image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                    image =  cv2.putText(image, f"Test-{randint(0, 100)}", (200, 1000), cv2.FONT_HERSHEY_COMPLEX, 12, (0, 0, 255), 2)
                    self.image_queue.put((ts, image))
                    logger.info(f"camera[{self.serial_number}] generate one image")
            else:
                time.sleep(0.01)


if __name__ == '__main__':
    import cv2
    def callback(ts, image):
        logger.info(f"received image {ts}")
        cv2.imwrite('test.jpg', image)  

    camera = FakeCamera(serial_number='Fake0001', image_width=640, image_height=480)
    camera.initialize()
    camera.add_image_received_callback(callback)
    camera.start_grab()
    for i in range(10):
        camera.trigger()
        time.sleep(0.5)
    time.sleep(1)
    camera.stop_grab()