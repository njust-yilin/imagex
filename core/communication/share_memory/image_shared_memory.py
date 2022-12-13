import numpy as np
from multiprocessing import shared_memory, Lock
from typing import Tuple
from loguru import logger

from core.utils import get_timestamp_ms


class ImageSharedMemory(object):
    def __init__(self, name:str, image_hwc:Tuple[int, int, int], image_count:int):
        self.image_count = image_count
        self.name = name
        self.image_hwc = image_hwc

        self.lock = Lock()
        self.index = 0

        # shared memory for images
        size = int(image_count * np.prod(image_hwc))
        self.images_shared_memory = self.init_shared_memory(name+'_image', size)
        self.shape = [image_count] + list(image_hwc)
        logger.info(f"Created {self.name} with shape {self.shape}")
        self.images = np.ndarray(self.shape, dtype=np.uint8, buffer=self.images_shared_memory.buf)

        # shared memory for image use count
        self.use_count_shared_memory = self.init_shared_memory(name+'_use_count', 4)
        self.use_count = np.ndarray((1,), dtype=np.uint32, buffer=self.use_count_shared_memory.buf)
        self.use_count[0] = 0

    def init_shared_memory(self, name, size):
        logger.info(f"Initializing shared memory {name}, size={size}")
        try:
            sm = shared_memory.SharedMemory(name=name, size=size, create=False)
            return sm
        except FileNotFoundError as e:
            logger.warning(e)
            sm = shared_memory.SharedMemory(name=name, size=size, create=True)
            return sm
        except FileExistsError as e:
            logger.warning(e)
            return sm

    @logger.catch
    def free(self, index:int):
        if index < 0 or index >= self.image_count:
            logger.error("Index out of range")
            return
        
        with self.lock:
            empty_image = np.zeros(self.image_hwc, dtype=np.uint8)
            self.images[index] = empty_image.data
            self.use_count[0] -= 1
            logger.info(f"Free image-{index}[{self.name}], use count={self.use_count[0]}")

    @logger.catch
    def free_all(self):
        with self.lock:
            empty_images = np.zeros(self.shape, dtype=np.uint8)
            self.images = empty_images.data
            logger.info(f"Free all images[{self.name}]")
            self.use_count[0] = 0

    def __del__(self):
        self.images_shared_memory.close()
        self.use_count_shared_memory.close()
        logger.info(f"Closed images shared memory {self.name}")


class ImageSharedMemoryProducer(ImageSharedMemory):
    def __init__(self, name:str, image_hwc:Tuple[int, int, int], image_count:int):
        super().__init__(name, image_hwc, image_count)
        self.free_all()

    def put(self, image:np.ndarray)->int:
        if self.use_count[0] >= self.image_count:
            logger.warning("Image shared memory full")
        
        ts = get_timestamp_ms()
        current_index =self.index
        with self.lock:
            self.images[self.index] = image.data
            self.index = (self.index + 1) % (self.image_count-1)
            self.use_count[0] += 1
            logger.info(f"Put image-{self.index}[{self.name}], use count={self.use_count[0]}, next_index={self.index}, took: {get_timestamp_ms()-ts}ms")
        return current_index

    def __del__(self):
        super().__del__()
        try:
            self.images_shared_memory.unlink()
            logger.info(f"Unlinked images shared memory {self.name}")
        except: pass
        try:
            self.use_count_shared_memory.unlink()
            logger.info(f"Unlinked use count shared memory {self.name}")
        except: pass


class ImageSharedMemoryConsumer(ImageSharedMemory):
    def __init__(self, name:str, image_hwc:Tuple[int, int, int], image_count:int):
        super().__init__(name, image_hwc, image_count)

    def get(self, index:int) -> np.ndarray:
        if index < 0 or index >= self.image_count:
            raise ValueError("Index out of range")
        
        ts = get_timestamp_ms()
        with self.lock:
            image = self.images[index]
            logger.info(f"Get image-{self.name}-{index}, took: {get_timestamp_ms()-ts}ms")
            return image

