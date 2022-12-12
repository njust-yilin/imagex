from imagex.api.rpc.imagex import imagex_pb2_grpc
from imagex.settings import configs
import grpc

def get_imagex_stub()->imagex_pb2_grpc.ImagexStub:
    return imagex_pb2_grpc.ImagexStub(grpc.insecure_channel(f'localhost:{configs.IMAGEX_RPC_PORT}'))


def get_ui_stub()->imagex_pb2_grpc.UIStub:
    return imagex_pb2_grpc.UIStub(grpc.insecure_channel(f'localhost:{configs.UI_RPC_PORT}'))



def get_deepx_stub()->imagex_pb2_grpc.DeepxStub:
    return imagex_pb2_grpc.DeepxStub(grpc.insecure_channel(f'localhost:{configs.DEEPX_RPC_PORT}'))



def get_lightx_stub()->imagex_pb2_grpc.LightxStub:
    return imagex_pb2_grpc.LightxStub(grpc.insecure_channel(f'localhost:{configs.LIGHTX_RPC_PORT}'))
