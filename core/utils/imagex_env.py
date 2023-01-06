from pathlib import Path


def _get_sub_home(directory)->Path:
    dir: Path = IMAGEX_HOME / directory
    dir.mkdir(parents=True, exist_ok=True)
    return dir


USER_HOME = Path.home()
IMAGEX_HOME = USER_HOME / 'imagex_data'

CONFIG_DIR = _get_sub_home('config')
NETWORK_DIR = _get_sub_home('networks')
TEMP_DIR = _get_sub_home('tmp')
PRETRAINED_MODELS_DIR = _get_sub_home('pretrained_models')
OUTPUT_DIR = _get_sub_home('output')
LOG_DIR = _get_sub_home('log')