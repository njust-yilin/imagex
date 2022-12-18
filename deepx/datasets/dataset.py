
import paddle
import os
from enum import Enum
import numpy as np

from core.utils import imagex_env
from deepx.cvlibs import manager
from deepx.transforms import Compose
import deepx.transforms.functional as F

class MODE(Enum):
    TRAIN='train'
    VAILD='valid'


@manager.DATASETS.add_component
class Dataset(paddle.io.Dataset):
    def __init__(
        self,
        mode,
        dataset_root,
        transforms,
        img_channels:int=3,
        sep=' ',
        ignore_index:int=255,
        edge=False) -> None:

        self.mode = MODE(mode)
        self.edge = edge
        if not os.path.exists(dataset_root):
            dataset_root = os.path.join(imagex_env.NETWORK_HOME, dataset_root)
        if not os.path.exists(dataset_root):
            raise FileNotFoundError(dataset_root)

        assert isinstance(transforms, list), f"Transforms is necessary, but got {transforms}"
        
        with open(os.path.join(dataset_root, 'labels.txt'), 'r') as f:
            classes = f.read().strip().splitlines()
            print(classes)
            self.num_classes = len(classes)
        
        if self.num_classes < 1:
            raise ValueError("num_classes should be at least 2, but got {num_classes}")
        
        if img_channels not in [1, 3]:
            raise ValueError(f"img_channels must be 1 or 3, but got {img_channels}")

        if self.mode == MODE.TRAIN:
            path = os.path.join(dataset_root, 'train_list.txt')
        else:
            path = os.path.join(dataset_root, 'val_list.txt')
        self.file_list = self._get_file_list(dataset_root, path, sep)

        self.mode = mode
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        self.img_channels = img_channels
        self.ignore_index = ignore_index

    def _get_file_list(self, dataset_root, path, sep):
        with open(path, 'r') as f:
            lines = f.read().strip().splitlines()
        
        file_list = []
        for line in lines:
            files = line.split(sep=sep)
            if len(files) != 2:
                print(f'File list should be image label, but got! {line}')
                continue
            image_file = os.path.join(dataset_root, files[0])
            mask_file = os.path.join(dataset_root, files[1])
            if not os.path.exists(image_file) or not os.path.exists(mask_file):
                print(f'File not found! {line}')
                continue
            file_list.append((image_file, mask_file))
        return file_list
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        data = {}
        data['train_info'] = []
        data['gt_fields'] = []
        data['img'], data['label'] = self.file_list[index]

        if self.mode == MODE.VAILD:
            data = self.transforms(data)
            data['label'] = data['label'][np.newaxis, :, :]
        else:
            data['gt_fields'].append('label')
            data = self.transforms(data)
            if self.edge:
                edge_mask = F.mask_to_binary_edge(
                    data['label'], radius=2, num_classes=self.num_classes)
                data['edge'] = edge_mask
        return data


if __name__ == '__main__':
    dataset = Dataset(
        'train',
        '/home/imagex/imagex_data/networks/optic_disc_seg',
        [])
    print(dataset.file_list[:1])
    print(dataset.num_classes)
    data = dataset[0]
    print(data['img'].shape)
    print(data['label'].shape)