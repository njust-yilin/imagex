from multiprocessing import Process, Event
from loguru import logger
import time

from core.utils.exceptions import ExitException

class Service(Process):
    def __init__(self):
        super().__init__()
        self._stop_event = Event()

    def setup(self):
        logger.info(f"Setup service: {self.name}")

    def routine(self):
        logger.info(f"routine service: {self.name}")
        time.sleep(1)

    def cleanup(self):
        logger.info(f"Cleanup service: {self.name}")

    def exit(self):
        logger.info(f"Exit service: {self.name}")
        self._stop_event.set()

    def run(self):
        self.setup()
        logger.info(f"Service started[{self.name}]")
        while not self._stop_event.is_set():
            try:
                self.routine()
            except ExitException as e:
                logger.error(e)
                break
            except Exception as e:
                logger.error(e)
                time.sleep(1)
        self.cleanup()
