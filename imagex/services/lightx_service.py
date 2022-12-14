from loguru import logger
import time
import numpy as np
import grpc
import time

# from imagex.pipe_elements import UIQueueElement
from imagex.services import Service
from imagex.settings import configs
from lightx.device.camera.fake_camera import FakeCamera
from imagex.api.rpc import imagex_pb2, imagex_pb2_grpc
from core.utils import get_timestamp_ms
from core.utils.grpc_helper import start_grpc_server, create_rpc_stub
from core.communication import ImageSharedMemoryClient


class LightxService(Service, imagex_pb2_grpc.LightxServicer):
    def __init__(self):
        Service.__init__(self)

        # 连接 rpc 服务器
        channel = grpc.insecure_channel(f'localhost:{configs.UI_RPC_PORT}')
        self.ui_rpc_stub = imagex_pb2_grpc.UIStub(channel)
        logger.info(f'Connected to localhost:{configs.UI_RPC_PORT}')

    def setup(self):
        # start rpc server
        self.server = start_grpc_server(self, imagex_pb2_grpc.add_LightxServicer_to_server, configs.LIGHTX_RPC_PORT)

        self.fake_camera = FakeCamera('Fake001', 2448, 2048)
        self.fake_camera.add_image_received_callback(self.on_image_ready)
    
    def cleanup(self):
        self.server.stop(0)
        self.fake_camera.stop_grab()
        if self.fake_camera:
            del self.fake_camera
        
        logger.info('Cleanup Lightx shared memory')
        if self.images_client:
            del self.images_client
        if self.masks_client:
            del self.masks_client
        return super().cleanup()
    
    def routine(self):
        time.sleep(.2)
        self.fake_camera.trigger()

    def on_image_ready(self, ts:int, image:np.ndarray):
        ts = get_timestamp_ms()
        image_index = self.images_client.put(image)
        mask_index = self.masks_client.put(image)
        response = self.ui_rpc_stub.UpdateImage(imagex_pb2.UpdateImageRequest(
            image_index=image_index,
            mask_index=mask_index,
        ))
        logger.info(f'got response{response} took {get_timestamp_ms()-ts}ms')

    # =================RPC API =================
    def Ping(self, request, context):
        return imagex_pb2.SuccessReply(ok=True)
    
    def Exit(self, request, context):
        self._stop_event.set()
        return imagex_pb2.Empty()

    def Initialize(self, request, context):
        # TODO: create camera's ImageSharedMemory
        logger.info('Initializing Lightx shared memory')
        self.images_client = ImageSharedMemoryClient(configs.IMAGEX_IMAGE_IMAGE_NAME, (2048, 2448, 3), 20)
        self.masks_client = ImageSharedMemoryClient(configs.IMAGEX_MASK_IMAGE_NAME, (2048, 2448, 3), 20)
        self.fake_camera.start_grab()
        return imagex_pb2.SuccessReply()


def get_service():
    return LightxService()