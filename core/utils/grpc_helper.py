import grpc
from concurrent import futures
from loguru import logger
from grpc._server import _Server


@logger.catch
def start_grpc_server(servicer, register, port:int)->_Server:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    register(servicer, server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    logger.info(f"RPC server started on port {port}")
    return server


@logger.catch
def create_rpc_stub(stub, port:int, ip:str='localhost'):
    return stub(grpc.insecure_channel(f'{ip}:{port}'))
