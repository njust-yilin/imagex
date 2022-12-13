from loguru import logger
import time, sys

from imagex.services import Service
from imagex.settings import configs
from imagex.api.rpc.imagex import imagex_pb2_grpc, imagex_pb2
from core.utils.grpc_helper import start_grpc_server, create_rpc_stub
from core.communication import ImageSharedMemoryManager


class ImagexService(Service, imagex_pb2_grpc.ImagexServicer):
    def __init__(self):
        # Service.__init__(self)
        super().__init__()

    def setup(self):
        super().setup()
        self.server = start_grpc_server(self, imagex_pb2_grpc.add_ImagexServicer_to_server, configs.IMAGEX_RPC_PORT)

        # TODO: create camera's ImageSharedMemory
        self.images_manager = ImageSharedMemoryManager(configs.IMAGEX_IMAGE_IMAGE_NAME, (2048, 2448, 3), 20)
        self.masks_manager = ImageSharedMemoryManager(configs.IMAGEX_MASK_IMAGE_NAME, (2048, 2448, 3), 20)

        return super().setup()
    
    def cleanup(self):
        try:
            deepx_stub: imagex_pb2_grpc.DeepxStub = create_rpc_stub(imagex_pb2_grpc.DeepxStub, configs.DEEPX_RPC_PORT)
            deepx_stub.Exit(imagex_pb2.Empty())
        except: pass

        try:
            deepx_stub: imagex_pb2_grpc.LightxStub = create_rpc_stub(imagex_pb2_grpc.LightxStub, configs.LIGHTX_RPC_PORT)
            deepx_stub.Exit(imagex_pb2.Empty())
        except:pass

        try:
            deepx_stub: imagex_pb2_grpc.UIStub = create_rpc_stub(imagex_pb2_grpc.UIStub, configs.UI_RPC_PORT)
            deepx_stub.Exit(imagex_pb2.Empty())
        except:pass

        # TODO: stop when all rpc termination
        time.sleep(1)
        del self.masks_manager
        del self.images_manager
        self.server.stop(0)
        sys.exit(0)
        super().cleanup()
    
    # ======================RPC API =================
    def Ping(self, request, context):
        return imagex_pb2.SuccessReply(ok=True)

    def InferReady(self, request, context):
        return imagex_pb2.SuccessReply(ok=True)

    def Exit(self, request, context):
        self._stop_event.set()
        return imagex_pb2.Empty()


def get_service():
    return ImagexService()