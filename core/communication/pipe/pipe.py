import os
from loguru import logger
from threading import Lock
import json

from imagex.settings import configs

class PipeCloseException(Exception):
    pass


class Pipe(object):
    def __init__(self, element_cls:object, namespace:str):
        name = f'{namespace}_{element_cls.__name__}.pipe'
        self.path = os.path.join(configs.IMAGEX_TMP_DIR, name)
        self.lock = Lock()
        self.pipe = None
        self.element_cls = element_cls

        try:
            os.mkfifo(self.path)
            logger.info(f"created pipe {self.path}")
        except: pass
        
    
    def __del__(self):
        logger.info(f"cleanup up pipe {self.path}")
        try:
            os.close(self.fifo)
        except: pass


class PipeConsumer(Pipe):
    def __init__(self, element_cls:object, namespace:str=''):
        super().__init__(element_cls, namespace)
        self.pipe = os.open(self.path, os.O_RDONLY)
    
    @logger.catch
    def get(self):
        buf = os.read(self.pipe, 65535)
        messages = buf.decode('utf-8').strip()
        if len(messages) == 0:
            raise PipeCloseException(f"{self.path} already closed")

        dicts = [json.loads(msg) for msg in messages.split('\n')]
        logger.info(dicts)
        instances = [self.element_cls(**dict) for dict in dicts]
        return instances

    def __del__(self):
        super().__del__()
        try:
            os.remove(self.path)
        except: pass


class PipeProducer(Pipe):
    def __init__(self, element_cls:object, namespace:str=''):
        super().__init__(element_cls, namespace)
        self.pipe = os.open(self.path, os.O_SYNC | os.O_CREAT | os.O_RDWR)

    def put(self, instance:object):
        assert isinstance(instance, self.element_cls)
        message = json.dumps(instance.__dict__) + '\n'
        os.write(self.pipe, message.encode('utf-8'))


if __name__ == '__main__':
    import time
    import numpy as np
    class Test(object):
        def __init__(self, name, index=2):
            self.name = name
            self.index = index
    
    element = Test('hello world')
    w = PipeProducer(Test, 'hello')
    r = PipeConsumer(Test, 'hello')

    w.put(element)
    w.put(element)
    logger.info(r.get())
    time.sleep(3)
    # del r
    # del w
