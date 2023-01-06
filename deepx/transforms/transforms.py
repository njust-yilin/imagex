import random
import cv2
import math
import numpy as np
from PIL import Image
from functools import reduce

from deepx.transforms import functional as F


KEY_FIELDS = 'trans_fields'


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data:dict):
        assert 'img' in data

        data['img'] = cv2.imread(data['img'])
        assert isinstance(data['img'], np.ndarray)
        if data['img'].ndim == 3:
            # BGR to RGB
            data['img'] = cv2.cvtColor(data['img'], cv2.COLOR_BGR2RGB)
        
        data['label'] = np.asarray(Image.open(data['label']))
        # 保存图像shape， 用于推理和验证
        if 'trans_info' not in data.keys():
            data['trans_info'] = []
        
        for trans in self.transforms:
            data = trans(data)

        if data['img'].ndim == 2:
            # HW To CHW
            data['img'] = data['img'][None, :, :]
        else:
            # HWC to CHW
            data['img'] = np.transpose(data['img'], (2, 0, 1))
        return data


class RandomHorizontalFlip:
    """随机水平翻转"""
    def __init__(self, prob=0.5) -> None:
        self.prob = prob
    
    def __call__(self, data:dict) -> dict:
        if random.random() < self.prob:
            data['img'] = F.horizontal_flip(data['img'])
            for key in data.get(KEY_FIELDS, []):
                data[key] = F.horizontal_flip(data[key])
        return data


class RandomVerticalFlip:
    """随机垂直翻转"""
    def __init__(self, prob=0.5) -> None:
        self.prob = prob
    
    def __call__(self, data:dict) -> dict:
        if random.random() < self.prob:
            data['img'] = F.vertical_flip(data['img'])
            for key in data.get(KEY_FIELDS, []):
                data[key] = F.vertical_flip(data[key])
        return data


class Resize:
    """尺寸修改"""
     # The interpolation mode
    interp_dict = {
        'NEAREST': cv2.INTER_NEAREST,
        'LINEAR': cv2.INTER_LINEAR,
        'CUBIC': cv2.INTER_CUBIC,
        'AREA': cv2.INTER_AREA,
        'LANCZOS4': cv2.INTER_LANCZOS4
    }
    def __init__(self, target_size=(512, 512), size_divisor=None, interp='LINEAR') -> None:
        self.target_size = target_size
        self.size_divisor = size_divisor
        self.interp = interp
    
    def __call__(self, data:dict) -> dict:
        data['trans_info'].append(('resize', data['img'].shape[0:2]))
        if self.interp == "RANDOM":
            interp = random.choice(list(self.interp_dict.values()))
        else:
            interp = self.interp_dict[self.interp]
        
        h, w = data['img'].shape[:2]
        target_size, _ = F.rescale_size((w, h), self.target_size, self.size_divisor)
        
        data['img'] = F.resize(data['img'], target_size, interp)
        for key in data.get(KEY_FIELDS, []):
            data[key] = F.resize(data[key], target_size, interp)
        return data
    

class ResizeRangeScaling:
    """随机尺寸缩放"""
    def __init__(self, min_value=400, max_value=600):
        assert max_value >= min_value
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, data:dict)->dict:
        if self.min_value == self.max_value:
            random_size = self.max_value
        else:
            random_size = int(np.random.uniform(self.min_value, self.max_value + .5))
        
        data['img'] = F.resize_long(data['img'], random_size, cv2.INTER_LINEAR)
        for key in data.get(KEY_FIELDS, []):
            data[key] = F.resize_long(data[key], random_size, cv2.INTER_NEAREST)
        return data


class ResizeStepScaling:
    """随机比例缩放"""
    def __init__(self, min_factor=0.75, max_factor=1.25, scale_step=0.25):
        assert max_factor >= min_factor
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.scale_step = scale_step

    def __call__(self, data:dict)->dict:
        if self.min_factor == self.max_factor:
            scale_factor = self.max_factor
        elif self.scale_step == 0:
            scale_factor = np.random.uniform(self.min_factor, self.max_factor)
        else:
            num_steps = int((self.max_factor - self.min_factor) / self.scale_step + 1)
            scale_factors = np.linspace(self.min_factor, self.max_factor, num_steps).tolist()
            scale_factor = random.choice(scale_factors)
        
        w = int(round(scale_factor * data['img'].shape[1]))
        h = int(round(scale_factor * data['img'].shape[0]))
        
        data['img'] = F.resize(data['img'], (w, h), cv2.INTER_LINEAR)
        for key in data.get(KEY_FIELDS, []):
            data[key] = F.resize(data[key], (w, h), cv2.INTER_NEAREST)
        return data


class Normallize:
    """图片归一化"""
    def __init__(self, mean=(0.5,), std=(0.5,)):
        assert isinstance(mean, (list, tuple))
        assert isinstance(std, (list, tuple))
        assert len(mean) in [1, 3]
        assert len(std) in [1,3]

        self.mean = np.array(mean)
        self.std = np.array(std)
        if reduce(lambda x, y: x * y, self.std) == 0:
            raise ValueError('{}: std is invalid!'.format(self))
    
    def __call__(self, data:dict)->dict:
        data['img'] = F.normalize(data['img'], self.mean, self.std)
        return data


class Padding:
    """添加边框"""
    def __init__(self, target_size, im_padding=127.5, label_padding=255) -> None:

        self.target_size = target_size
        self.im_padding = im_padding
        self.label_padding = label_padding
    
    def __call__(self, data:dict)->dict:
        h, w = data['img'].shape[:2]
        data['trans_info'].append(('padding', (h, w)))

        if isinstance(self.target_size, int):
            target_w = self.target_size
            target_h = self.target_size
        else:
            target_w, target_h = self.target_size
        
        pad_h = target_h - h
        pad_w = target_w - w
        assert pad_h >= 0 and pad_w >= 0

        img_channels = 1 if data['img'].ndim == 2 else data['img'].shape[2]

        data['img'] = cv2.copyMakeBorder(
                data['img'],
                0,
                pad_h,
                0,
                pad_w,
                cv2.BORDER_CONSTANT,
                value=(self.im_padding, ) * img_channels)
        for key in data.get(KEY_FIELDS, []):
            data[key] = cv2.copyMakeBorder(
                data[key],
                0,
                pad_h,
                0,
                pad_w,
                cv2.BORDER_CONSTANT,
                value=self.label_padding)
        return data


class PaddingByAspectRatio:
    """添加边框"""
    def __init__(self, aspect_ratio=1, im_padding=127.5, label_padding=255) -> None:

        self.aspect_ratio = aspect_ratio
        self.im_padding = im_padding
        self.label_padding = label_padding
    
    def __call__(self, data:dict)->dict:
        h, w = data['img'].shape[:2]
        
        ratio = w / h
        if ratio == self.aspect_ratio:
            return data
        elif ratio > self.aspect_ratio:
            h = int(w / self.aspect_ratio)
        else:
            w = int(h * self.aspect_ratio)
        
        padding = Padding((w, h), self.im_padding, self.label_padding)
        return padding(data)


class RandomPaddingCrop:
    """随机裁剪"""
    def __init__(self, crop_size=(512, 512), im_padding=127.5, label_padding=255):
        self.crop_size = crop_size
        self.im_padding = im_padding
        self.label_padding = label_padding

    def __call__(self, data:dict)->dict:
        if isinstance(self.crop_size, int):
            crop_w = self.crop_size
            crop_h = self.crop_size
        else:
            crop_w, crop_h = self.crop_size
        
        img_h, img_w = data['img'].shape[:2]
        if img_h == crop_h and img_w == crop_w:
            return data
        
        pad_h = max(crop_h - img_h, 0)
        pad_w = max(crop_w - img_w, 0)
        img_channels = 1 if data['img'].ndim == 2 else data['img'].shape[2]

        # 裁剪后尺寸比原尺寸大，添加边框
        if (pad_h > 0 or pad_w > 0):
            data['img'] = cv2.copyMakeBorder(
                data['img'],
                0,
                pad_h,
                0,
                pad_w,
                cv2.BORDER_CONSTANT,
                value=(self.im_padding, ) * img_channels)
            for key in data.get(KEY_FIELDS, []):
                data[key] = cv2.copyMakeBorder(
                    data[key],
                    0,
                    pad_h,
                    0,
                    pad_w,
                    cv2.BORDER_CONSTANT,
                    value=self.label_padding)
            # 裁剪后的尺寸
            img_h, img_w = data['img'].shape[:2]

        # 裁剪
        if crop_h > 0 and crop_w > 0:
            # 随机裁剪开始位置
            h_off = np.random.randint(img_h - crop_h + 1)
            w_off = np.random.randint(img_w - crop_w + 1)

            if data['img'].ndim == 2:
                data['img'] = data['img'][h_off:(crop_h + h_off),
                                            w_off:(w_off + crop_w)]
            else:
                data['img'] = data['img'][h_off:(crop_h + h_off),
                                            w_off:(w_off + crop_w), :]
            for key in data.get(KEY_FIELDS, []):
                data[key] = data[key][h_off:(crop_h + h_off), w_off:(
                    w_off + crop_w)]
        return data


class RandomCenterCrop:
    """随机中心裁剪"""
    def __init__(self, retain_ratio=(0.5, 0.5)):
        self.retain_ratio = retain_ratio

    def __call__(self, data:dict)->dict:
        retain_width, retain_height = self.retain_ratio
        img_height, img_width = data['img'].shape[:2]

        if retain_height == 1. and retain_width == 1.:
            return data

        # 随机裁剪尺寸
        rand_w = np.random.randint(img_width * (1 - retain_width))
        rand_h = np.random.randint(img_height * (1 - retain_height))
        # 随机裁剪起始
        offset_w = 0 if rand_w == 0 else np.random.randint(rand_w)
        offset_h = 0 if rand_w == 0 else np.random.randint(rand_h)
        
        p0 = offset_h
        p1 = img_height + offset_h - rand_h
        p2 = offset_w
        p3 = img_height + offset_w - rand_w
        if data['img'].ndim == 2:
            data['img'] = data['img'][p0:p1, p2:p3]
        else:
            data['img'] = data['img'][p0:p1, p2:p3]
        for key in data.get(KEY_FIELDS, []):
            data[key] = data[key][p0:p1, p2:p3]
        return data


class ScalePadding:
    """中心添加边框并修改尺寸"""
    def __init__(self, target_size=(512, 512), im_padding=127.5, label_padding=255):
        self.target_size = target_size
        self.im_padding = im_padding
        self.label_padding = label_padding

    def __call__(self, data:dict)->dict:
        h, w = data['img'].shape[:2]
        image_channels = 1 if data['img'].ndim == 2 else data['img'].shape[2]

        # 创建方图
        if data['img'].ndim == 2:
            new_im = np.zeros((max(h, w), max(h, w))) + self.im_padding
        else:
            new_im = np.zeros((max(h, w), max(h, w), image_channels)) + self.im_padding

        if 'label' in data[KEY_FIELDS]:
            new_label = np.zeros((max(h, w), max(h, w))) + self.label_padding
        
        # 粘贴图片
        if h > w:
            padding = int((h - w) / 2)
            if data['img'].ndim == 2:
                new_im[:, padding:padding + w] = data['img']
            else:
                new_im[:, padding:padding + w, :] = data['img']
            if 'label' in data[KEY_FIELDS]:
                new_label[:, padding:padding + w] = data['label']
        else:
            padding = int((w - h) / 2)
            if data['img'].ndim == 2:
                new_im[padding:padding + h, :] = data['img']
            else:
                new_im[padding:padding + h, :, :] = data['img']
            if 'label' in data[KEY_FIELDS]:
                new_label[padding:padding + h, :] = data['label']
        
        data['img'] = F.resize(np.uint8(new_im), self.target_size, cv2.INTER_CUBIC)
        if 'label' in data[KEY_FIELDS]:
            data['label'] = F.resize(np.uint8(new_label), self.target_size, cv2.INTER_CUBIC)
        return data


class RandomNoise:
    """随机噪声"""
    def __init__(self, prob=0.5, max_sigma=10.0) -> None:
        self.prob = prob
        self.max_sigma = max_sigma
    
    def __call__(self, data:dict)->dict:
        if random.random() < self.prob:
            mu = 0
            sigma = random.random() * self.max_sigma
            data['img'] = np.array(data['img'], dtype=np.float32)
            data['img'] += np.random.normal(mu, sigma, data['img'].shape)
            data['img'][data['img'] > 255] = 255
            data['img'][data['img'] < 0] = 0
        return data


class RandomBlur:
    """随机模糊"""
    def __init__(self, prob=0.1, blur_type="gaussian"):
        self.prob = prob
        self.blur_type = blur_type
    
    def __call__(self, data:dict)->dict:
        if self.prob <= 0:
            n = 0
        elif self.prob >= 1:
            n = 1
        else:
            n = int(1.0 / self.prob)

        if n <= 0 or np.random.randint(0, n) != 0:
            return data
        
        radius = np.random.randint(3, 10)
        if radius % 2 != 1:
            radius += 1
        if radius > 9:
            radius = 9
        data['img'] = np.array(data['img'], dtype=np.uint8)

        if self.blur_type == 'random':
            blur_type = np.random.choice(['median', 'blur', 'gaussian'])
        else:
            blur_type = self.blur_type

        if blur_type == "median":
            data['img'] = cv2.medianBlur(data['img'], radius)
        elif blur_type == 'blur':
            data['img'] = cv2.blur(data['img'], (radius, radius))
        else:
            data['img'] = cv2.GaussianBlur(data['img'], (radius, radius), 0, 0)

        data['img'] = np.array(data['img'], dtype=np.float32)
        return data


class RandomRotation:
    """随机旋转"""
    def __init__(self, max_rotation=15, im_padding=127.5, label_padding=255) -> None:
        self.max_rotation = max_rotation
        self.im_padding = im_padding
        self.label_padding = label_padding

    def __call__(self, data:dict)->dict:
        if self.max_rotation <= 0:
            return data
        
        h, w = data['img'].shape[:2]
        img_channels = 1 if data['img'].ndim == 2 else data['img'].shape[2]
        do_rotation = np.random.uniform(-self.max_rotation, self.max_rotation)

        pc = (w // 2, h // 2)
        r = cv2.getRotationMatrix2D(pc, do_rotation, 1.0)
        cos = np.abs(r[0, 0])
        sin = np.abs(r[0, 1])

        nw = int((h * sin) + (w * cos))
        nh = int((h * cos) + (w * sin))

        (cx, cy) = pc
        r[0, 2] += (nw / 2) - cx
        r[1, 2] += (nh / 2) - cy
        dsize = (nw, nh)
        data['img'] = cv2.warpAffine(
            data['img'],
            r,
            dsize=dsize,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(self.im_padding, ) * img_channels)
        for key in data.get(KEY_FIELDS, []):
            data[key] = cv2.warpAffine(
                data[key],
                r,
                dsize=dsize,
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=self.label_padding)
        return data


class RandomScaleAspect:
    """随机横纵比"""
    def __init__(self, min_scale=0.5, aspect_ratio=0.33):
        self.min_scale = min_scale
        self.aspect_ratio = aspect_ratio
    
    def __call__(self, data:dict)->dict:
        if self.min_scale == 0 or self.aspect_ratio == 0:
            return data
        
        h, w = data['img'].shape[:2]
        for i in range(10):
            area = w * h
            target_area = area * np.random.uniform(self.min_scale, 1.0)
            aspect_ratio = np.random.uniform(self.aspect_ratio, 1.0/self.aspect_ratio)

            dw = int(np.sqrt(target_area * 1.0 * aspect_ratio))
            dh = int(np.sqrt(target_area * 1.0 / aspect_ratio))
            if (np.random.randint(10) < 5):
                dw, dh = dh, dw
            
            if dh < h and dw < w:
                h1 = np.random.randint(0, h - dh)
                w1 = np.random.randint(0, w - dh)
                if data['img'].ndim == 2:
                    data['img'] = data['img'][h1:(h1 + dh), w1:(w1 + dw)]
                else:
                    data['img'] = data['img'][h1:(h1 + dh), w1:(w1 + dw), :]
                data['img'] = cv2.resize(data['img'], (w, h), interpolation=cv2.INTER_LINEAR)
                for key in data.get(KEY_FIELDS, []):
                    data[key] = data[key][h1:(h1 + dh), w1:(w1 + dw)]
                    data[key] = cv2.resize(data[key], (w, h), interpolation=cv2.INTER_NEAREST)
                break
        return data


class RandomDistort:
    """随机构建图像"""
    def __init__(self,
                 brightness_range=0.5,
                 brightness_prob=0.5,
                 contrast_range=0.5,
                 contrast_prob=0.5,
                 saturation_range=0.5,
                 saturation_prob=0.5,
                 hue_range=18,
                 hue_prob=0.5,
                 sharpness_range=0.5,
                 sharpness_prob=0):
        self.brightness_range = brightness_range
        self.brightness_prob = brightness_prob
        self.contrast_range = contrast_range
        self.contrast_prob = contrast_prob
        self.saturation_range = saturation_range
        self.saturation_prob = saturation_prob
        self.hue_range = hue_range
        self.hue_prob = hue_prob
        self.sharpness_range = sharpness_range
        self.sharpness_prob = sharpness_prob

    def __call__(self, data:dict)->dict:
        brightness_lower = 1 - self.brightness_range
        brightness_upper = 1 + self.brightness_range
        contrast_lower = 1 - self.contrast_range
        contrast_upper = 1 + self.contrast_range
        saturation_lower = 1 - self.saturation_range
        saturation_upper = 1 + self.saturation_range
        hue_lower = -self.hue_range
        hue_upper = self.hue_range
        sharpness_lower = 1 - self.sharpness_range
        sharpness_upper = 1 + self.sharpness_range
        ops = [F.brightness, F.contrast, F.saturation, F.sharpness]
        if data['img'].ndim > 2:
            ops.append(F.hue)
        random.shuffle(ops)

        params_dict = {
            'brightness': {
                'low': brightness_lower,
                'high': brightness_upper
            },
            'contrast': {
                'low': contrast_lower,
                'high': contrast_upper
            },
            'saturation': {
                'low': saturation_lower,
                'high': saturation_upper
            },
            'hue': {
                'low': hue_lower,
                'high': hue_upper
            },
            'sharpness': {
                'low': sharpness_lower,
                'high': sharpness_upper,
            }
        }
        prob_dict = {
            'brightness': self.brightness_prob,
            'contrast': self.contrast_prob,
            'saturation': self.saturation_prob,
            'hue': self.hue_prob,
            'sharpness': self.sharpness_prob
        }
        data['img'] = data['img'].astype('uint8')
        data['img'] = Image.fromarray(data['img'])
        for id in range(len(ops)):
            params = params_dict[ops[id].__name__]
            prob = prob_dict[ops[id].__name__]
            params['im'] = data['img']
            if np.random.uniform(0, 1) < prob:
                data['img'] = ops[id](**params)
        data['img'] = np.asarray(data['img']).astype('float32')
        return data


class RandomAffine:
    def __init__(self,
                 size=(224, 224),
                 translation_offset=0,
                 max_rotation=15,
                 min_scale_factor=0.75,
                 max_scale_factor=1.25,
                 im_padding_value=128,
                 label_padding_value=255):
        self.size = size
        self.translation_offset = translation_offset
        self.max_rotation = max_rotation
        self.min_scale_factor = min_scale_factor
        self.max_scale_factor = max_scale_factor
        self.im_padding_value = im_padding_value
        self.label_padding_value = label_padding_value

    def __call__(self, data:dict)->dict:

        w, h = self.size
        bbox = [0, 0, data['img'].shape[1] - 1, data['img'].shape[0] - 1]
        x_offset = (random.random() - 0.5) * 2 * self.translation_offset
        y_offset = (random.random() - 0.5) * 2 * self.translation_offset
        dx = (w - (bbox[2] + bbox[0])) / 2.0
        dy = (h - (bbox[3] + bbox[1])) / 2.0

        matrix_trans = np.array([[1.0, 0, dx], [0, 1.0, dy], [0, 0, 1.0]])

        angle = random.random() * 2 * self.max_rotation - self.max_rotation
        scale = random.random() * (self.max_scale_factor - self.min_scale_factor
                                   ) + self.min_scale_factor
        scale *= np.mean(
            [float(w) / (bbox[2] - bbox[0]), float(h) / (bbox[3] - bbox[1])])
        alpha = scale * math.cos(angle / 180.0 * math.pi)
        beta = scale * math.sin(angle / 180.0 * math.pi)

        centerx = w / 2.0 + x_offset
        centery = h / 2.0 + y_offset
        matrix = np.array(
            [[alpha, beta, (1 - alpha) * centerx - beta * centery],
             [-beta, alpha, beta * centerx + (1 - alpha) * centery],
             [0, 0, 1.0]])

        matrix = matrix.dot(matrix_trans)[0:2, :]
        img_channels = 1 if data['img'].ndim == 2 else data['img'].shape[2]
        data['img'] = cv2.warpAffine(
            np.uint8(data['img']),
            matrix,
            tuple(self.size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(self.im_padding_value, ) * img_channels)
        for key in data.get(KEY_FIELDS, []):
            data[key] = cv2.warpAffine(
                np.uint8(data[key]),
                matrix,
                tuple(self.size),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=self.label_padding_value)
        return data