from loguru import logger
import os
import paddle.nn as nn
import paddle
from pathlib import Path
import numpy as np

from core.utils import imagex_env 

def worker_init_fn(worker_id):
    np.random.seed(np.random.randint(0, 100000))

def load_entire_model(model:nn.Layer, pretrained:str):
    if pretrained is None:
        logger.warning("No pretrained model to load")
        return
    return load_pretrained_model(model, pretrained)

def load_pretrained_model(model:nn.Layer, pretrained:str):
    pretrained:Path = imagex_env.PRETRAINED_MODELS_DIR / f'{pretrained}.pdparams'
    if not os.path.exists(pretrained):
        logger.warning(f"File not found: {pretrained}")
        pretrained

    para_state_dict = paddle.load(pretrained.as_posix())
    model_state_dict = model.state_dict()
    num_params_loaded = 0
    for key in model_state_dict.keys():
        if key not in para_state_dict:
            logger.warning(f"{key} not found in pretrained model")
        elif list(para_state_dict[key].shape) != list(model_state_dict[key].shape):
            logger.warning(f"{key} shape not match in pretrained model")
        else:
            model_state_dict[key] = para_state_dict[key]
            num_params_loaded += 1
    model.set_dict(model_state_dict)
    logger.info("There are {}/{} variables loaded into {}.".format(
                num_params_loaded,
                len(model_state_dict), model.__class__.__name__))


if __name__ == "__main__":
    model = nn.Linear(10, 10)
    load_entire_model(model, 'PP_STDCNet2')