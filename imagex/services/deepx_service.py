from loguru import logger
import time

from imagex.services import Service
from imagex.settings import configs
from imagex.api.rpc.imagex import imagex_pb2_grpc, imagex_pb2
from core.utils.grpc_helper import start_grpc_server
from core.communication import ImageSharedMemoryClient


class DeepxService(Service, imagex_pb2_grpc.DeepxServicer):
    def __init__(self):
        Service.__init__(self)

    def setup(self):
        self.server = start_grpc_server(self, imagex_pb2_grpc.add_DeepxServicer_to_server, configs.DEEPX_RPC_PORT)

        return super().setup()
    
    def cleanup(self):
        logger.info('Cleanup Deepx shared memory')
        if self.images_client:
            del self.images_client
        if self.masks_client:
            del self.masks_client
        self.server.stop(0)
        return super().cleanup()
    
    # ====================== RPC API =================
    def Ping(self, request, context):
        return imagex_pb2.SuccessReply(ok=True)

    def Infer(self, request, context):
        return super().Infer(request, context)

    def Exit(self, request, context):
        self._stop_event.set()
        return imagex_pb2.Empty()
    
    def Initialize(self, request, context):
        # TODO: create camera's ImageSharedMemory
        logger.info('Initializing Deepx shared memory')
        self.images_client = ImageSharedMemoryClient(configs.IMAGEX_IMAGE_IMAGE_NAME, (2048, 2448, 3), 20)
        self.masks_client = ImageSharedMemoryClient(configs.IMAGEX_MASK_IMAGE_NAME, (2048, 2448, 3), 20)
        return imagex_pb2.SuccessReply(ok=True)


def get_service():
    return DeepxService()