from loguru import logger
import time

from imagex.services import Service
from imagex.settings import configs
from imagex.api.rpc.imagex import imagex_pb2_grpc, imagex_pb2
from core.utils.grpc_help import create_rpc_server


class DeepxService(Service, imagex_pb2_grpc.DeepxServicer):
    def __init__(self):
        Service.__init__(self)

    def setup(self):
        self.start_rpc_server()
        return super().setup()
    
    def cleanup(self):
        time.sleep(0.1)
        self.server.stop(0)
        return super().cleanup()

    def start_rpc_server(self):
        self.server = create_rpc_server(configs.DEEPX_RPC_PORT)
        imagex_pb2_grpc.add_DeepxServicer_to_server(self, self.server)
        self.server.start()
        logger.info(f"{self.name} RPC server started on port {configs.DEEPX_RPC_PORT}")
    
    def Ping(self, request, context):
        return imagex_pb2.SuccessReply(ok=True)

    def Infer(self, request, context):
        return super().Infer(request, context)

    def Exit(self, request, context):
        self._stop_event.set()
        return imagex_pb2.Empty()


def get_service():
    return DeepxService()