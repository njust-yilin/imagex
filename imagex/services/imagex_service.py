from loguru import logger
import time

from imagex.services import Service
from imagex.settings import configs
from imagex.api.rpc.imagex import imagex_pb2_grpc, imagex_pb2
from core.utils.grpc_helper import start_grpc_server, create_rpc_stub


class ImagexService(Service, imagex_pb2_grpc.ImagexServicer):
    def __init__(self):
        # Service.__init__(self)
        super().__init__()

    def setup(self):
        super().setup()
        self.server = start_grpc_server(self, imagex_pb2_grpc.add_ImagexServicer_to_server, configs.IMAGEX_RPC_PORT)
        return super().setup()
    
    def cleanup(self):
        deepx_stub: imagex_pb2_grpc.DeepxStub = create_rpc_stub(imagex_pb2_grpc.DeepxStub, configs.DEEPX_RPC_PORT)
        deepx_stub.Exit(imagex_pb2.Empty())

        deepx_stub: imagex_pb2_grpc.LightxStub = create_rpc_stub(imagex_pb2_grpc.LightxStub, configs.LIGHTX_RPC_PORT)
        deepx_stub.Exit(imagex_pb2.Empty())

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