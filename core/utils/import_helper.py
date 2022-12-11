import importlib
from loguru import logger


@logger.catch
def load_module(module: str):
    return importlib.import_module(module)