import tempfile
import contextlib
from urllib.parse import urlparse, unquote
import filelock
import os
import paddle
from paddle import nn
from paddle.optimizer import Optimizer
import numpy as np
import random

from deepx.utils import logger
from core.utils import imagex_env
from deepx.utils.download import download_file_and_uncompress


@contextlib.contextmanager
def generate_tmpdir(directory:str=None, **kwargs):
    '''Generate a temporary directory'''
    directory = imagex_env.TEMP_HOME if not directory else directory
    with tempfile.TemporaryDirectory(dir=directory, **kwargs) as _dir:
        yield _dir

def load_entire_model(model:nn.Layer, pretrained):
    if pretrained is not None:
        load_pretrained_model(model, pretrained)
    else:
        logger.warning('Not all pretrained params of {} are loaded, ' \
                       'training from scratch or a pretrained backbone.'.format(model.__class__.__name__))


def load_pretrained_model(model:nn.Layer, pretrained_model):
    if pretrained_model is None:
        logger.info(f'No pretrained model to load, {model.__class__.__name__} will be trained from scratch.')

    logger.info(f'Loading pretrained model from {pretrained_model}')
    if urlparse(pretrained_model).netloc:
        pretrained_model = download_pretrained_model(pretrained_model)
    else:
        pretrained_model = imagex_env.get_pretrained_model_path(pretrained_model)

    if not os.path.exists(pretrained_model):
        raise ValueError(f'The pretrained model directory is not Found: {pretrained_model}')

    para_state_dict = paddle.load(pretrained_model)
    model_state_dict = model.state_dict()
    keys = model_state_dict.keys()
    num_params_loaded = 0

    for k in keys:
        if k not in para_state_dict:
            logger.warning("{k} is not in pretrained model")
        elif list(para_state_dict[k].shape) != list(model_state_dict[k].shape):
            logger.warning(
                f"[SKIP] Shape of pretrained params {k} doesn't match.(Pretrained: {para_state_dict[k].shape}, Actual: {model_state_dict[k].shape}")
        else:
            model_state_dict[k] = para_state_dict[k]
            num_params_loaded += 1
    model.set_dict(model_state_dict)
    logger.info(f"There are {num_params_loaded}/{len(model_state_dict)} variables loaded into {model.__class__.__name__}.")


def download_pretrained_model(pretrained_model):
    assert urlparse(pretrained_model).netloc, "The url is not valid."
    pretrained_model = unquote(pretrained_model)
    savename = pretrained_model.split('/')[-1]
    if not savename.endswith(('tgz', 'tar.gz', 'tar', 'zip')):
        savename = pretrained_model.split('/')[-2]
        filename = pretrained_model.split('/')[-1]
    else:
        savename = savename.split('.')[0]
        filename = 'model.pdparams'
    logger.info(f'Downloading model: {pretrained_model}')
    with generate_tmpdir() as _dir:
        with filelock.FileLock(os.path.join(imagex_env.TEMP_HOME, savename)):
            pretrained_model = download_file_and_uncompress(
                pretrained_model,
                savepath=_dir,
                extrapath=imagex_env.PRETRAINED_MODELS_HOME,
                extraname=savename,
                filename=filename
            )
            pretrained_model = os.path.join(pretrained_model, filename)
    return pretrained_model

def resume(model:nn.Layer, optimizer:Optimizer, resume_model):
    if resume_model is not None:
        logger.info(f'Resume model from {resume_model}')
        if os.path.exists(resume_model):
            resume_model = os.path.normpath(resume_model)
            ckpt_path = os.path.join(resume_model, 'model.pdparams')
            para_state_dict = paddle.load(ckpt_path)
            ckpt_path = os.path.join(resume_model, 'model.pdopt')
            opti_state_dict = paddle.load(ckpt_path)
            model.set_state_dict(para_state_dict)
            optimizer.set_state_dict(opti_state_dict)
            epoch = resume_model.split('_')[-1]
            epoch = int(epoch)
            return epoch
        else:
            raise ValueError('Directory of the model needed to resume is not Found: {resume_model}')
    else:
        logger.info('No model needed to resume.')


def worker_init_fn(worker_id):
    np.random.seed(random.randint(0, 100000))


if __name__ == '__main__':
    import paddle.vision.models
    import paddle.optimizer.adam
    model = paddle.vision.models.AlexNet()
    pretrained_model = load_pretrained_model(model, 'PP_STDCNet2')
    # model = paddle.vision.models.AlexNet()
    # print(model)