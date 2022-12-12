from loguru import logger
import time

from imagex.services import Service
from imagex.settings import configs
from imagex.api.rpc.imagex import imagex_pb2_grpc, imagex_pb2
from imagex.api.rpc import stub_helper
from core.utils.grpc_help import create_rpc_server


class ImagexService(Service, imagex_pb2_grpc.ImagexServicer):
    def __init__(self):
        # Service.__init__(self)
        super().__init__()

    def setup(self):
        super().setup()
        self.start_rpc_server()
        return super().setup()

    def start_rpc_server(self):
        self.server = create_rpc_server(configs.IMAGEX_RPC_PORT)
        imagex_pb2_grpc.add_ImagexServicer_to_server(self, self.server)
        self.server.start()
        logger.info(f"{self.name} RPC server started on port {configs.IMAGEX_RPC_PORT}")
    
    def cleanup(self):
        stub_helper.get_deepx_stub().Exit(imagex_pb2.Empty())
        stub_helper.get_lightx_stub().Exit(imagex_pb2.Empty())
        self.server.stop(0)
        super().cleanup()
    
    # ======================RPC API =================
    def Ping(self, request, context):
        return imagex_pb2.SuccessReply(ok=True)

    def InferReady(self, request, context):
        return imagex_pb2.SuccessReply(ok=True)

    def Exit(self, request, context):
        self._stop_event.set()
        return imagex_pb2.Empty()

    def DeepxReady(self, request, context):
        return imagex_pb2.SuccessReply(ok=True)

    
    def UIReady(self, request, context):
        return imagex_pb2.SuccessReply(ok=True)

    
    def LightxReady(self, request, context):
        return imagex_pb2.SuccessReply(ok=True)


def get_service():
    return ImagexService()