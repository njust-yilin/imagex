from torch.utils.data import Dataset
import torchvision.transforms
from pathlib import Path
import os
from typing import Tuple, List
from PIL import Image
import numpy as np

from deepx.transforms import KEY_FIELDS
from deepx.utils import is_image


class SegDataset(Dataset):
    def __init__(self, root, mode='train', transforms=None):
        self.transforms = transforms
        self.mode = mode

        root = Path(root) / mode
        if not root.is_dir():
            raise ValueError(f'{root} is not a directory')
        
        self.file_list = []

        images_path = root / 'images'
        masks_path = root / 'masks'
        for file in os.listdir(images_path):
            mask_path = masks_path / file
            image_path = images_path / file
            if is_image(file) and mask_path.is_file():
                self.file_list.append((image_path.as_posix(), mask_path.as_posix()))
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        data = {}
        data['img'], data['label'] = self.file_list[index]
        data[KEY_FIELDS] = []
        if self.mode == 'valid':
            data = self.transforms(data)
            data['label'] = data['label'][None, :, :] # To CHW
        else:
            data[KEY_FIELDS].append('label')
            data = self.transforms(data)
        return data


if __name__ == '__main__':
    import numpy as np
    from torch.utils.data import DataLoader
    from deepx import transforms
    root = Path.home() / 'imagex_data/networks/optic_disc_seg'
    dataset = SegDataset(root, transforms=transforms.Compose(
        transforms.Resize((512, 512)),
        transforms.Normallize()
    ))
    data = dataset[0]
    image, label = data['img'], data['label']
    print(label.shape)
    print(image.shape)
    data_loader = DataLoader(dataset=dataset, batch_size=2)
    for data in data_loader:
        image, label = data['img'], data['label']
        print(label.shape)
        print(image.shape)
        break
