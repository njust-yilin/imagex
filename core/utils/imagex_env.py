from pathlib import Path
import os


def _get_imagex_home():
    imagex_home = os.getenv('IMAGEX_HOME', 'imagex_data')
    if os.path.exists(imagex_home):
        return imagex_home
    return os.path.join(Path.home(), 'imagex_data')


def _get_sub_home(directory):
    home = os.path.join(_get_imagex_home(), directory)
    if not os.path.exists(home):
        os.makedirs(home, exist_ok=True)
    return home


def get_pretrained_model_path(name:str):
    return os.path.join(PRETRAINED_MODELS_HOME, name, 'model.pdparams')


USER_HOME = Path.home()
NETWORK_HOME = _get_sub_home('networks')
TEMP_HOME = _get_sub_home('tmp')
PRETRAINED_MODELS_HOME = _get_sub_home('pretrained_models')
