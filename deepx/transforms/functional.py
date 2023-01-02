import cv2
import numpy as np
from PIL import Image, ImageEnhance
import math


def normalize(im, mean, std):
    """标准差归一化"""
    im = im.astype(np.float32, copy=False) / 255.0
    im -= mean
    im /= std
    return im


def rescale_size(img_size, target_size, size_divisor=None):
    """保持长宽比缩放"""
    scale = max(target_size) / max(img_size), min(target_size) / max(img_size)
    scale = min(scale)
    rescaled_size = [round(i * scale) for i in img_size]

    if size_divisor:
        rescaled_size = [
            math.ceil(i / size_divisor) * size_divisor
            for i in rescaled_size
        ]
    return rescaled_size, scale


def resize(im:np.ndarray, targer_size=608, interp=cv2.INTER_LINEAR):
    """修改尺寸"""
    if isinstance(targer_size, list) or isinstance(targer_size, tuple):
        w, h = targer_size[0], targer_size[1]
    else:
        w, h = targer_size, targer_size
    im = cv2.resize(im, (w, h), interpolation=interp)
    return im


def resize_long(im:np.ndarray, long_size=224, interp=cv2.INTER_LINEAR):
    """修改尺寸"""
    scale = float(long_size) / max(im.shape[0], im.shape[1])
    h, w = [int(round(i * scale)) for i in im.shape[:2]]
    return cv2.resize(im, (w, h), interpolation=interp)


def resize_short(im:np.ndarray, short_size=224, interp=cv2.INTER_LINEAR):
    """修改尺寸"""
    scale = float(short_size) / min(im.shape[0], im.shape[1])
    h, w = [int(round(i * scale)) for i in im.shape[:2]]
    return cv2.resize(im, (w, h), interpolation=interp)


def vertical_flip(im:np.ndarray):
    """垂直翻转"""
    if im.ndim == 3:
        im = im[:, ::-1, :]
    elif im.ndim == 2:
        im = im[:, ::-1]
    return im

def horizontal_flip(im:np.ndarray):
    """水平翻转"""
    if im.ndim == 3:
        im = im[::-1, :, :]
    elif im.ndim == 2:
        im = im[::-1, :]
    return im


def brightness(im, low, high):
    """随机亮度变化"""
    delta = np.random.uniform(low, high)
    im = ImageEnhance.Brightness(im).enhance(delta)
    return im


def contrast(im, low, high):
    """随即对比度变化"""
    delta = np.random.uniform(low, high)
    return ImageEnhance.Contrast(im).enhance(delta)


def saturation(im, low, high):
    """随机饱和度变换"""
    delta = np.random.uniform(low, high)
    return ImageEnhance.Color(im).enhance(delta)


def hue(im, low, high):
    """随机色彩变换"""
    delta = np.random.uniform(low, high)
    im = np.array(im.convert('HSV'))
    im[:, :, 0] = im[:, :, 0] + delta
    return Image.fromarray(im, mode='HSV').convert('RGB')


def sharpness(im, low, high):
    """随机锐度变换"""
    delta = np.random.uniform(low, high)
    return ImageEnhance.Sharpness(im).enhance(delta)


def rotate(im, low, high):
    """随机旋转变换"""
    delta = np.random.uniform(low, high)
    return im.rotate(int(delta))