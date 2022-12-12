from loguru import logger
import time
import numpy as np
import grpc

# from imagex.pipe_elements import UIQueueElement
from core.communication import ImageSharedMemory
from imagex.services import Service
from imagex.settings import configs
from lightx.device.camera.fake_camera import FakeCamera
from imagex.api.rpc.ui import ui_pb2, ui_pb2_grpc
from core.utils import get_timestamp_ms


class LightXService(Service):
    def __init__(self):
        Service.__init__(self)

        # 连接 rpc 服务器
        channel = grpc.insecure_channel(f'localhost:{configs.UI_RPC_PORT}')
        self.ui_rpc_stub = ui_pb2_grpc.UIStub(channel)
        logger.info(f'Connected to localhost:{configs.UI_RPC_PORT}')

    def setup(self):
        # TODO: create camera's ImageSharedMemory
        self.images_producer = ImageSharedMemory(configs.IMAGEX_IMAGE_IMAGE_NAME, (2048, 2448, 3), 20)
        self.masks_producer = ImageSharedMemory(configs.IMAGEX_MASK_IMAGE_NAME, (2048, 2448, 3), 20)
        # self.ui_queue = PipeProducer(UIQueueElement, configs.UI_QUEUE_NAME)

        self.fake_camera = FakeCamera('Fake001', 2448, 2048)
        self.fake_camera.add_image_received_callback(self.on_image_ready)
        time.sleep(2)
        self.fake_camera.start_grab()
    
    def routine(self):
        time.sleep(.2)
        self.fake_camera.trigger()

    def on_image_ready(self, ts:int, image:np.ndarray):
        ts = get_timestamp_ms()
        image_index = self.images_producer.put(image)
        mask_index = self.masks_producer.put(image)
        response = self.ui_rpc_stub.ImageUpdate(ui_pb2.ImageUpdateRequest(
            image_index=image_index,
            mask_index=mask_index,
        ))
        logger.info(f'got response{response} took {get_timestamp_ms()-ts}ms')

        # element = UIQueueElement(
        #     image_index=image_index,
        #     mask_index=mask_index,
        #     image_hwc=(2048, 2448, 3),
        #     timestamp=ts,
        #     camera_id=self.fake_camera.serial_number,
        #     part_id='',
        #     part_timestamp=0,
        #     network='test',
        #     capture_config='test',
        #     capture_index=0,
        #     ok=True
        # )
        # self.ui_queue.put(element)

def get_service():
    return LightXService()