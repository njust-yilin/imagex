from torch.utils.data import Dataset
import torchvision.transforms
from pathlib import Path
import os
from typing import Tuple, List
from PIL import Image
import numpy as np

from deepx.utils import is_image


class SegDataset(Dataset):
    def __init__(self, root, mode='train', transforms=None):
        if transforms is None:
            self.transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()
            ])
        elif isinstance(transforms, List) or isinstance(transforms, List):
            self.transforms = torchvision.transforms.Compose(transforms)
        else:
            self.transforms = transforms

        root = Path(root) / mode
        if not root.is_dir():
            raise ValueError(f'{root} is not a directory')
        
        self.image_files = []
        self.mask_files = []

        images_path = root / 'images'
        masks_path = root / 'masks'
        for file in os.listdir(images_path):
            if is_image(file):
                self.image_files.append((images_path / file).as_posix())
        
        for file in os.listdir(masks_path):
            if is_image(file):
                self.mask_files.append((masks_path / file).as_posix())
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image = Image.open(self.image_files[index]).convert('RGB')
        label = Image.open(self.mask_files[index])

        image = self.transforms(image)
        label = self.transforms(label)

        return image, label


if __name__ == '__main__':
    import numpy as np
    from torch.utils.data import DataLoader
    from torchvision import transforms
    root = Path.home() / 'imagex_data/networks/optic_disc_seg'
    dataset = SegDataset(root)
    data_loader = DataLoader(dataset=dataset, batch_size=1)
    for data in data_loader:
        image, label = data
        print(label.shape)
        print(image.shape)
        break
