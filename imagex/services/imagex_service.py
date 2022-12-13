from loguru import logger
import time

from imagex.services import Service
from imagex.settings import configs
from imagex.api.rpc.imagex import imagex_pb2_grpc, imagex_pb2
from core.utils.grpc_helper import start_grpc_server, create_rpc_stub
from core.utils.exception_helper import run_catch_except
from core.communication import ImageSharedMemoryManager


class ImagexService(Service, imagex_pb2_grpc.ImagexServicer):
    def __init__(self):
        # Service.__init__(self)
        super().__init__()

    def setup(self):
        self.server = start_grpc_server(self, imagex_pb2_grpc.add_ImagexServicer_to_server, configs.IMAGEX_RPC_PORT)

        # TODO: create camera's ImageSharedMemory
        logger.info('Initializing Imagex shared memory')
        self.images_manager = ImageSharedMemoryManager(configs.IMAGEX_IMAGE_IMAGE_NAME, (2048, 2448, 3), 20)
        self.masks_manager = ImageSharedMemoryManager(configs.IMAGEX_MASK_IMAGE_NAME, (2048, 2448, 3), 20)

        # wait for lightx, ui and deepx service to start
        self.wait_for_service_start(imagex_pb2_grpc.UIStub, configs.UI_RPC_PORT, "UI Service")
        self.wait_for_service_start(imagex_pb2_grpc.DeepxStub, configs.DEEPX_RPC_PORT, "Deepx Service")
        self.wait_for_service_start(imagex_pb2_grpc.LightxStub, configs.LIGHTX_RPC_PORT, "Lightx Service")

        return super().setup()
    
    def wait_for_service_start(self, stub_class, port, name):
        while not self._stop_event.is_set():
            stub = create_rpc_stub(stub_class, port)
            response = run_catch_except(stub.Ping, args=[imagex_pb2.Empty()])
            if response:
                run_catch_except(stub.Initialize, args=[imagex_pb2.Empty()])
                break
            logger.warning(f"Wait for service:{name}-{port} to start...")
            time.sleep(1)


    def heartbeat(self):
        pass
    
    def cleanup(self):
        stub:imagex_pb2_grpc.DeepxStub = create_rpc_stub(imagex_pb2_grpc.DeepxStub, configs.DEEPX_RPC_PORT)
        run_catch_except(stub.Exit, args=[imagex_pb2.Empty()])

        stub:imagex_pb2_grpc.UIStub = create_rpc_stub(imagex_pb2_grpc.UIStub, configs.UI_RPC_PORT)
        run_catch_except(stub.Exit, args=[imagex_pb2.Empty()])

        stub:imagex_pb2_grpc.LightxStub = create_rpc_stub(imagex_pb2_grpc.LightxStub, configs.LIGHTX_RPC_PORT)
        run_catch_except(stub.Exit, args=[imagex_pb2.Empty()])

        # TODO: stop when all rpc termination
        time.sleep(1)
        logger.info('Cleanup Imagex shared memory')
        if self.masks_manager:
            del self.masks_manager
        if self.images_manager:
            del self.images_manager
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


def get_service():
    return ImagexService()