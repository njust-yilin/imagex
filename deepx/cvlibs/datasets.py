from pathlib import Path
import os
import paddle
import numpy as np

from deepx.transforms import KEY_FIELDS, Compose
from deepx.utils import is_image
from deepx.cvlibs import manager


@manager.DATASETS.add_component
class SegDataset(paddle.io.Dataset):
    def __init__(self, root, mode='train', transforms=[]):
        self.transforms = Compose(transforms)
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
        data['trans_info'] = []
        data['img'], data['label'] = self.file_list[index]

        data[KEY_FIELDS] = []
        
        # If key in gt_fields, the data[key] have transforms synchronous.
        data[KEY_FIELDS] = ['label']
        data = self.transforms(data)
        if self.mode == 'valid':
            data['label'] = data['label'][np.newaxis, :, :]
        
        return data


if __name__ == '__main__':
    from deepx import transforms
    from paddle.io import DataLoader
    root = Path.home() / 'imagex_data/networks/optic_disc_seg'
    dataset = manager.DATASETS['SegDataset'](root, transforms=[
        transforms.Resize((512, 512)),],
        mode='valid'
    )

    data_loader =  DataLoader(dataset=dataset, batch_size=2)
    for data in data_loader:
        image, label = data['img'], data['label']
        print(label.shape)
        print(image.shape)
