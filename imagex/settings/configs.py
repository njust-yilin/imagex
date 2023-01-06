import os
from pathlib import Path


# path
def get_path(path:Path, name:str):
    path.mkdir(parents=True, exist_ok=True)
    return path.joinpath(name)

HOME = Path.home()
IMAGEX_HOME = get_path(HOME, 'imagex_data')
IMAGEX_INSTALL_DIR = get_path(HOME, 'imagex')

# imagex_data children dir
IMAGEX_CONFIG_DIR = get_path(IMAGEX_HOME, 'config')
IMAGEX_TMP_DIR = get_path(IMAGEX_HOME, 'tmp')
IMAGEX_MODELS_DIR = get_path(IMAGEX_HOME, 'models')
IMAGEX_LOGS_DIR = get_path(IMAGEX_HOME, 'logs')
IMAGEX_NETWORKS_DIR = get_path(IMAGEX_HOME, 'networks')
IMAGEX_OUTPUTS_DIR = get_path(IMAGEX_HOME, 'outputs')
IMAGEX_IMAGESS_DIR = get_path(IMAGEX_HOME, 'images')

LICENSE_PATH = get_path(IMAGEX_HOME, 'LICENSE')
SERVICE_CONFIG_PATH = get_path(IMAGEX_CONFIG_DIR, "service.json")
IMAGEX_DB_PATH = get_path(IMAGEX_CONFIG_DIR, "imagex.db")

# networs path
NETWORK_IMAGES_DIR_NAME = os.getenv('NETWORK_IMAGES_DIR_NAME', 'images')
NETWORK_LABELS_DIR_NAME = os.getenv('NETWORK_LABELS_DIR_NAME', 'labels')
NETWORK_MASKS_DIR_NAME = os.getenv('NETWORK_LABELS_DIR_NAME', 'masks')
NETWORK_RESULTS_DIR_NAME = os.getenv('NETWORK_RESULTS_DIR_NAME', 'results')
NETWORK_TRAIN_DIR_NAME = os.getenv('NETWORK_TRAIN_DIR_NAME', 'train')
NETWORK_VALIDATE_DIR_NAME = os.getenv('NETWORK_VALIDATE_DIR_NAME', 'validate')

# env
IMAGEX_LOG_PRIFIX = os.getenv('IMAGEX_LOG_PRIFIX', 'imagex')
DEEPX_LOG_PRIFIX = os.getenv('DEEPX_LOG_PRIFIX', 'deepx')
LIGHTX_LOG_PRIFIX = os.getenv('LIGHTX_LOG_PRIFIX', 'lightx')
IMAGEX_VERSION = os.getenv('IMAGEX_VERSION', 'V0.1.1')
IMAGEX_NAME = os.getenv('IMAGEX_NAME', 'ImageX')
IMAGEX_ICON_PATH = os.getenv('IMAGEX_ICON_PATH', 'imagex/assets/icon/icon.png')

IMAGEX_MASK_IMAGE_NAME = os.getenv('IMAGEX_MASK_IMAGE_NAME', 'mask')
IMAGEX_IMAGE_IMAGE_NAME = os.getenv('IMAGEX_MASK_IMAGE_NAME', 'image')
UI_QUEUE_NAME = os.getenv('UI_QUEUE_NAME', 'ui')
IMAGEX_QUEUE_NAME = os.getenv('IMAGEX_QUEUE_NAME', 'imagex')
DEEPX_QUEUE_NAME = os.getenv('DEEPX_QUEUE_NAME', 'deepx')

IMAGEX_RPC_PORT = os.getenv('IMAGEX_RPC_PORT', 15000)
UI_RPC_PORT = os.getenv('UI_RPC_PORT', 15001)
LIGHTX_RPC_PORT = os.getenv('LIGHTX_RPC_PORT', 15002)
DEEPX_RPC_PORT = os.getenv('DEEPX_RPC_PORT', 15003)

