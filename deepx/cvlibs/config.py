from paddle import nn
import paddle.optimizer
import yaml

from deepx.cvlibs import manager
from deepx.utils import logger
from deepx.datasets import Dataset

KEY_BATCH_SIZE = 'batch_size'
KEY_EPOCH = 'epoch'
KEY_LR_SCHEDULER = 'lr_scheduler'
KEY_OPTIMIZER = 'optimizer'
KEY_MODEL= 'model'
KEY_NUM_CLASSES = 'num_classes'
KEY_TRAIN_DATASET = 'train_dataset'
KEY_VALID_DATASET = 'valid_dataset'


class Config(object):
    def __init__(
            self,
            config:dict
        ) -> None:
        self._config = config
        self._model: nn.Layer = None
        self._lr_scheduler: paddle.optimizer.lr.LRScheduler = None
        self._optimizer: paddle.optimizer.Optimizer = None
        self._losses = None
        self._train_dataset: Dataset = None
        self._valid_dataset: Dataset = None
    
    def _load_component(self, name: str):
        components = [
            manager.MODELS, manager.BACKBONES,
            manager.DATASETS, manager.TRANSFORMS,
            manager.LOSSES
        ]
        for component in components:
            if name in component.components_dict:
                return component[name]
        else:
            raise RuntimeError(f'The specified component was not found {name}.')
    
    def _load_object(self, cfg:dict):
        cfg = cfg.copy()
        if 'type' not in cfg:
            raise RuntimeError(f'No object information in {cfg}.')
        component = self._load_component(cfg.pop('type'))

        params = {}
        for key, value in cfg.items():
            if self._is_meta_type(value):
                params[key] = self._load_object(value)
            elif isinstance(value, list):
                params[key] = [
                    self._load_object(item)
                    if self._is_meta_type(item) else item
                    for item in value
                ]
            else:
                params[key] = value
        return component(**params)

    def _is_meta_type(self, item) -> bool:
        return isinstance(item, dict) and 'type' in item

    def __str__(self) -> str:
        return yaml.dump(self._config)
    
    def _prepare_loss(self, name)->dict:
        args = self._config.get(name, {}).copy()
        if 'types' in args and 'coef' in args:
            len_types = len(args['types'])
            len_coef = len(args['coef'])
            if len_types!= len_coef:
                if len_types == 1:
                    args['types'] = args['types'] * len_coef
                else:
                    raise ValueError(
                        f'The length of types should equal to coef or equal to 1 in loss config, \
                            but they are {len_types} and {len_coef}.')
        else:
            raise ValueError('Loss config should contain keys of "types" and "coef"')
        
        losses = dict()
        for key, value in args.items():
            if key == 'types':
                losses[key] = []
                for item in args[key]:
                    if item['type'] != 'MixedLoss':
                        if 'ignore_index' in item:
                            assert item['ignore_index'] == self.train_dataset.ignore_index, 'If ignore_index of loss is set, '\
                            'the ignore_index of loss and train_dataset must be the same. \nCurrently, loss ignore_index = {}, '\
                            'train_dataset ignore_index = {}. \nIt is recommended not to set loss ignore_index, so it is consistent with '\
                            'train_dataset by default.'.format(item['ignore_index'], self.train_dataset.ignore_index)
                        item['ignore_index'] = self.train_dataset.ignore_index
                    losses[key].append(self._load_object(item))
            else:
                losses[key] = value
        
        return losses

    @staticmethod
    def parse_from_yaml(path:str):
        with open(path, 'r') as file:
            dict = yaml.load(file, Loader=yaml.FullLoader)
        return Config(dict)
    
    @property
    def valid_transforms(self) -> list:
        return self.valid_dataset.transforms
    
    @property
    def loss(self) -> dict:
        if self._losses is None:
            self._losses = self._prepare_loss('loss')
        return self._losses
    
    @property
    def distill_loss(self) -> dict:
        if not hasattr(self, '_distill_losses'):
            self._distill_losses = self._prepare_loss('distill_loss')
        return self._distill_losses
    
    @property
    def train_dataset(self) -> Dataset:
        if self._train_dataset:
            return self._train_dataset
        if KEY_TRAIN_DATASET not in self._config:
            raise RuntimeError(f'train_dataset must be specified in config')

        cfg = self._config[KEY_TRAIN_DATASET]
        cfg['type'] = 'Dataset'
        self._train_dataset = self._load_object(cfg)
        return self._train_dataset

    @property
    def valid_dataset(self) -> Dataset:
        if self._valid_dataset:
            return self._valid_dataset
        if KEY_VALID_DATASET not in self._config:
            raise RuntimeError(f'valid_dataset must be specified in config')

        cfg = self._config[KEY_VALID_DATASET]
        cfg['type'] = 'Dataset'
        self._valid_dataset = self._load_object(cfg)
        return self._valid_dataset
    
    @property
    def batch_size(self) -> int:
        return self._config.get(KEY_BATCH_SIZE, 1)
    
    @property
    def epoch(self) -> int:
        return self._config.get(KEY_EPOCH, 500)
    
    @property
    def lr_scheduler(self) -> paddle.optimizer.lr.LRScheduler:
        if self._lr_scheduler:
            return self._lr_scheduler

        if KEY_LR_SCHEDULER not in self._config:
            raise RuntimeError('Not [lr_scheduler] configured in config file')

        params:dict = self._config.get('lr_scheduler', {}).copy()

        use_warmup = False
        if 'warmup_epoch' in params:
            use_warmup = True
            warmup_epoch = params.pop('warmup_epoch')
            assert 'warmup_start_lr' in params, \
                "When use warmup, please set warmup_start_lr and warmup_epoch in lr_scheduler"
            warmup_start_lr = params.pop('warmup_start_lr')
            end_lr = params['learning_rate']
        
        lr_type = params.pop('type')
        if lr_type == 'PolynomialDecay':
            epoch = self.epoch - warmup_epoch if use_warmup else self.epoch
            epoch = max(epoch, 1)
            params.setdefault('decay_steps', epoch)
            params.setdefault('end_lr', 0)
            params.setdefault('power', 0.9)
        lr_sche = getattr(paddle.optimizer.lr, lr_type)(**params)

        if use_warmup:
            lr_sche = paddle.optimizer.lr.LinearWarmup(
                learning_rate=lr_sche,
                warmup_steps=warmup_epoch,
                start_lr=warmup_start_lr,
                end_lr=end_lr)

        self._lr_scheduler = lr_sche
        return lr_sche
    
    @property
    def num_classes(self) -> int:
        if not KEY_NUM_CLASSES in self._config:
            raise RuntimeError('Not [num_classes] configured in config file')
        return self._config[KEY_NUM_CLASSES]

    @property
    def model(self) -> nn.Layer:
        if self._model:
            return self._model

        if KEY_MODEL not in self._config:
            raise RuntimeError('Not [model] configured in config file')

        cfg = self._config.get(KEY_MODEL, {})
        cfg[KEY_NUM_CLASSES] = self.num_classes
        self._model = self._load_object(cfg)
        return self._model

    
    @property
    def optimizer(self) -> paddle.optimizer.Optimizer:
        if self._optimizer:
            return self._optimizer

        if KEY_OPTIMIZER not in self._config:
            raise RuntimeError('Not [optimizer] configured in config file')

        lr = self.lr_scheduler
        args = self._config.get(KEY_OPTIMIZER, {}).copy()
        optimizer_type = args.pop('type')
        if optimizer_type == 'sgd':
            args.setdefault('momentum', 0.9)

        model_params = self.model.parameters()
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
                    'learning_rate': backbone_lr_mult
                }, {
                    'params': other_params
                }]

        if optimizer_type == 'sgd':
            self._optimizer = paddle.optimizer.Momentum(lr, parameters=model_params, **args)
        elif optimizer_type == 'adam':
            self._optimizer = paddle.optimizer.Adam(lr, parameters=model_params, **args)
        elif optimizer_type in paddle.optimizer.__all__:
            self._optimizer = getattr(paddle.optimizer, optimizer_type)(lr, parameters=model_params, **args)
        else:
            raise RuntimeError('Unknown optimizer type {}.'.format(optimizer_type))
        return self._optimizer


if __name__ == '__main__':
    path = '/home/imagex/Desktop/yilin/imagex/notbook/pp_liteseg_optic_disc_512x512_1k.yml'
    config = Config.parse_from_yaml(path)
    print(f'batch size: {config.batch_size}')
    print(f'epoch: {config.epoch}')
    print(f'lr_scheduler: {config.lr_scheduler}')
    print(f'model: {type(config.model)}')
    print(f'optimizer: {config.optimizer}')
    print(f'train_dataset: {config.train_dataset}')
    print(f'train_dataset: {config.train_dataset.transforms}')
    print(f'valid_dataset: {config.valid_dataset}')
    print(f'loss: {config.loss}')
    print(f'valid_transforms: {config.valid_transforms}')
    # print(config)