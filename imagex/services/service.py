from multiprocessing import Process, Event
from loguru import logger
import time

from core.utils.exceptions import ExitException

class Service(Process):
    def __init__(self):
        super().__init__()
        self._stop_event = Event()

    def setup(self):
        pass

    def routine(self):
        pass

    def exit(self):
        self._stop_event.set()

    def run(self):
        self.setup()
        logger.info(f"Starting service[{self.name}]")
        while not self._stop_event.is_set():
            try:
                self.routine()
            except ExitException as e:
                logger.error(e)
                break
            except Exception as e:
                logger.error(e)
            time.sleep(1)
