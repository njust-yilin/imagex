from PySide6.QtWidgets import QApplication
from loguru import logger
from PySide6 import QtGui
import grpc
from concurrent import futures

from imagex.services import Service
from imagex.settings import configs
from imagex.ui.main_window import MainWindow
from imagex.api.rpc.ui import ui_pb2, ui_pb2_grpc
from imagex.ui.ui_thread import UIThread


class UIService(Service, ui_pb2_grpc.UIServicer):
    def __init__(self):
        Service.__init__(self)
    
    def run(self):
        logger.info("Starting UIViewService")
        app = QApplication([configs.IMAGEX_NAME])
        app.setApplicationDisplayName(f"{configs.IMAGEX_NAME} {configs.IMAGEX_VERSION}")
        icon = QtGui.QIcon(configs.IMAGEX_ICON_PATH)
        app.setWindowIcon(icon)

        self.ui_thread = UIThread()
        win = MainWindow(self.ui_thread)

        # start rpc server
        self.start_rpc_server()

        logger.info("StartedUIViewService")
        app.exec()
        self.server.stop(0)
        del app

    def start_rpc_server(self):
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        ui_pb2_grpc.add_UIServicer_to_server(self, self.server)
        self.server.add_insecure_port(f'[::]:{configs.UI_RPC_PORT}')
        self.server.start()
        logger.info(f"RPC server started on port {configs.UI_RPC_PORT}")
    
    def Ping(self, request, context):
        return ui_pb2.SuccessReply(ok=True)

    def ImageUpdate(self, request:ui_pb2.ImageUpdateRequest, context):
        self.ui_thread.image_ready(request)
        return ui_pb2.SuccessReply(ok=True)

    def Exit(self, request, context):
        self.ui_thread.exit()
        return ui_pb2.Empty()


def get_service():
    return UIService()