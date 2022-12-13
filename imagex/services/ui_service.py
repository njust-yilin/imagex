from PySide6.QtWidgets import QApplication
from loguru import logger
from PySide6 import QtGui
import time

from imagex.services import Service
from imagex.settings import configs
from imagex.ui.main_window import MainWindow
from imagex.api.rpc.imagex import imagex_pb2_grpc, imagex_pb2
from imagex.ui.ui_thread import UIThread
from core.utils.grpc_helper import start_grpc_server
from core.utils import grpc_helper


class UIService(Service, imagex_pb2_grpc.UIServicer):
    def __init__(self):
        Service.__init__(self)
    
    def run(self):
        logger.info("Starting UIViewService")
        app = QApplication([configs.IMAGEX_NAME])
        app.setApplicationDisplayName(f"{configs.IMAGEX_NAME} {configs.IMAGEX_VERSION}")
        icon = QtGui.QIcon(configs.IMAGEX_ICON_PATH)
        app.setWindowIcon(icon)

        self.ui_thread = UIThread()
        self.ui_thread.exit_ui.connect(self.exit)
        win = MainWindow(self.ui_thread)

        # start rpc server
        self.server = start_grpc_server(self, imagex_pb2_grpc.add_UIServicer_to_server, configs.UI_RPC_PORT)

        logger.info(f"Started {self.name}")
        app.exec()
        del app
    
    def exit(self):
        logger.info("Notify Imagex Service stopped")
        grpc_helper.get_imagex_stub().Exit(imagex_pb2.Empty())
        # self.server.stop(0)
        return super().exit()

    # ===============================RPC API =============================
    def Ping(self, request, context):
        return imagex_pb2.SuccessReply(ok=True)

    def UpdateImage(self, request:imagex_pb2.UpdateImageRequest, context):
        self.ui_thread.image_ready(request)
        return imagex_pb2.SuccessReply(ok=True)

    def Exit(self, request, context):
        self.server.stop()
        return imagex_pb2.Empty()

    def Initialize(self, request, context):
        return imagex_pb2.SuccessReply(ok=True)


def get_service():
    return UIService()