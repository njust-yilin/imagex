from abc import ABCMeta, abstractmethod
import numpy as np
from threading import RLock, Event
from queue import Queue
import time
from loguru import logger


class AbstractCamera(metaclass=ABCMeta):
    def __init__(self, serial_number, image_width, image_height) -> None:
        self.serial_number:str = serial_number
        self.gain:int = 192
        self.pixel_format = 'Momo8'

        self.max_width:int = 0
        self.max_height:int = 0
        self.width:int = image_width
        self.height:int = image_height
        self.x_offset:int = 0
        self.y_offset:int = 0

        self.trigger_mode:bool = True
        self.exposure_us:float = 0
        self.trigger_source:str = 'Software'
        self.acquisition_status_selector:str = 'FrameTriggerWait'
        self.trigger_selector:str = 'FrameStart'
        self.exposure_mode:str = 'Timed'
        self.enable_acquisition:bool = True
        self.frame_rate:float = 0
        self.result_frame_rate:float = 0

        self.callback = None
        self.lock = RLock()
        self.stop_event = Event()
        self.image_queue = Queue()

    @staticmethod
    def get_cameras():
        pass
    
    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def start_grab(self):
        pass

    @abstractmethod
    def stop_grab(self):
        pass

    @abstractmethod
    def trigger(self):
        pass

    @abstractmethod
    def image_convert(self, image:np.ndarray)->np.ndarray:
        pass

    def async_process_images(self):
        while not self.stop_event.is_set():
            if not self.image_queue.empty():
                ts, image = self.image_queue.get()
                image = self.image_convert(image)
                self.execute_callback(ts=ts, image=image)
            else:
                time.sleep(0.001)
        logger.info(f"quitted image queue async_process_images")

    def execute_callback(self, **kwargs):
        if self.callback:
            try:
                self.callback(**kwargs)
            except Exception as e:
                logger.error(e)

    def add_image_received_callback(self, callback):
        self.callback = callback
