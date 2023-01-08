import yaml
from loguru import logger
import os
import paddle.nn as nn
import paddle.optimizer as optimizer
from pathlib import Path

from deepx.cvlibs import manager
from core.utils import imagex_env


class Config(object):
    def __init__(self, config:str) -> None:
        assert isinstance(config, dict), "config must be a dict"

        root = Path(config['root'])
        if not root.is_dir():
            if imagex_env.NETWORK_DIR.joinpath(root).is_dir():
                config['root'] = imagex_env.NETWORK_DIR.joinpath(root).as_posix()
            else:
                raise ValueError(f'{root} is not a directory')

        config['train_dataset']['root'] = config['root']
        config['valid_dataset']['root'] = config['root']
        self._config = config

        self._model = None
        self._losses = None

    @property
    def num_classes(self):
        return self._config['model']['num_classes']

    @property
    def root(self):
        return self._config['root']
    
    @property
    def batch_size(self) -> int:
        return self._config.get('batch_size', 1)
    
    @property
    def epochs(self) -> int:
        return self._config.get('epochs', 500)

    @property
    def model(self) -> nn.Layer:
        if self._model:
            return self._model
        model_cfg = self._config.get('model', {}).copy()
        self._model = Config.load_object(model_cfg)
        return self._model
    
    @property
    def lr_scheduler(self) -> optimizer.lr.LRScheduler:
        assert 'lr_scheduler' in self._config, \
            "No `lr_scheduler` specified in the config"

        params:dict = self._config.get('lr_scheduler', {}).copy()
        use_warmup = False
        if 'warmup_epochs' in params:
            use_warmup = True
            assert 'warmup_start_lr' in params and 'warmup_epochs' in params, \
                "When use warmup, please set warmup_start_lr and warmup_epochs in lr_scheduler"

            warmup_epochs = params.pop('warmup_epochs')
            warmup_start_lr = params.pop('warmup_start_lr')
            end_lr = params['learning_rate']

        lr_type = params.pop('type')
        if lr_type == 'PolynomialDecay':
            epochs = self.epochs - warmup_epochs if use_warmup else self.epochs
            iters = epochs * len(self.train_dataset)
            iters = max(iters, 1)
            params.setdefault('decay_steps', iters)
            params.setdefault('end_lr', 0)
            params.setdefault('power', 0.9)
        
        lr_sche = getattr(optimizer.lr, lr_type)(**params)

        if use_warmup:
            lr_sche = optimizer.lr.LinearWarmup(
                learning_rate=lr_sche,
                warmup_steps=warmup_epochs,
                start_lr=warmup_start_lr,
                end_lr=end_lr
            )
        return lr_sche
    
    @property
    def optimizer(self) -> optimizer.Optimizer:
        lr = self.lr_scheduler
        args = self._config.get('optimizer', {})
        if args['type'] == 'sgd':
            args.setdefault('momentum', 0.9)
        optimizer_type = args.pop('type')
        params = self.model.parameters()

        if 'backbone_lr_mult' in args:
            if not hasattr(self.model, 'backbone'):
                logger.warning('The backbone_lr_mult is not effective because'
                               ' the model does not have backbone')
            else:
                backbone_lr_mult = args.pop('backbone_lr_mult')
                backbone_params = self.model.backbone.parameters()
                backbone_params_id = [id(x) for x in backbone_params]
                other_params = [
                    x for x in params if id(x) not in backbone_params_id
                ]
                params = [{
                    'params': backbone_params,
                    'learning_rate': backbone_lr_mult,
                },{
                    'params': other_params
                }]
        
        if optimizer_type == 'sgd':
            obj = optimizer.Momentum
        elif optimizer_type == 'adam':
            obj = optimizer.Adam
        elif optimizer_type in optimizer.__all__:
            obj =  getattr(optimizer, optimizer_type)
        else:
            raise RuntimeError(f'Unknown optimizer type {optimizer_type}')
        return obj(lr, parameters=params, **args)
    
    @property
    def train_dataset(self):
        return Config.load_object(self._config['train_dataset'])

    @property
    def vaild_dataset(self):
        return Config.load_object(self._config['valid_dataset'])

    @property
    def loss(self) -> dict:
        if self._losses:
            return self._losses

        args = self._config.get('loss', {}).copy()
        if 'types' in args and 'coef' in args:
            len_types = len(args['types'])
            len_coef = len(args['coef'])
            if len_types != len_coef:
                if len_types == 1:
                    args['types'] = args['types'] * len_coef
                else:
                    raise ValueError('len_types must be equal len_coef')
        else:
            raise ValueError('types and coef must be specified in confg')
        
        types = [Config.load_object(item) for item in args['types']]
        self._losses = {'types':types, 'coef':args['coef']}

        return self._losses

    @staticmethod
    def is_meta_type(item):
        return isinstance(item, dict) and 'type' in item

    @staticmethod
    def load_object(cfg:dict):
        cfg = cfg.copy()
        if 'type' not in cfg:
            raise RuntimeError(f'No object type specified in {cfg}')
        component = Config.load_component(cfg.pop('type'))

        params = {}
        for key, value in cfg.items():
            if Config.is_meta_type(value):
                params[key] = Config.load_object(value)
            elif isinstance(value, list):
                params[key] = [
                    Config.load_object(item) if Config.is_meta_type(item) else item
                    for item in value
                ]
            else:
                params[key] = value
        return component(**params)
        
    
    @staticmethod
    def load_component(component_name: str):
        for components in manager.ComponentList:
            if component := components.get(component_name):
                return component
        else:
            raise RuntimeError(f'Component {component_name} not found')

    @staticmethod
    def parse_from_yaml(path:str):
        if not path:
            raise ValueError("path must be specified")

        if not os.path.exists(path):
            raise FileExistsError(f'File {path} not found')
        
        if not (path.endswith('yml') or path.endswith('yaml')):
            raise RuntimeError(f"File {path} must be a YAML file")
        
        with open(path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            if config.get('model', None) is None:
                raise RuntimeError('No model specified in config')

            return Config(config)


if __name__ == '__main__':
    path = '/home/imagex/Desktop/imagex/configs/pp_liteseg_optic_disc_512x512_1k.yml'
    config = Config.parse_from_yaml(path)
    print(config.epochs)
    print(config.batch_size)
    print(config.lr_scheduler)
    print(type(config.model))
    print(config.optimizer)
    print(config.loss)
    print(len(config.train_dataset))
    print(len(config.vaild_dataset))