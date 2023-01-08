import paddle.nn as nn
from collections.abc import Sequence
import paddle
import paddle.nn.functional as F


def reverse_transform(pred, trans_info, mode='nearest'):
    int_type_list = [paddle.int8, paddle.int16, paddle.int32, paddle.int64]
    dtype = pred.dtype
    for item in trans_info[::-1]:
        if isinstance(item[0], list):
            trans_mode = item[0][0]
        else:
            trans_mode = item[0]
        if trans_mode == 'resize':
            h, w = item[1][:2]
            if paddle.get_device() == 'cpu' and dtype in int_type_list:
                pred = paddle.cast(pred, 'float32')
                pred = F.interpolate(pred, [h, w], mode=mode)
                pred = paddle.cast(pred, dtype)
            else:
                pred = F.interpolate(pred, [h, w], mode=mode)
        elif trans_mode == 'padding':
            h, w = item[1][:2]
            pred = pred[:, :, 0:h, 0:w]
        else:
            raise Exception("Unexpected info '{}' in im_info".format(item[0]))
        return pred


def inference(model:nn.Layer, im):
    if hasattr(model, 'data_format') and model.data_format == 'NHWC':
        im = im.transpose((0, 2, 3, 1)) # NCHW to NHWC
    
    logits = model(im)
    logit = logits[0]
    assert isinstance(logits, Sequence), f"logits[type(logits)] must be a sequence"
    if hasattr(model, 'data_format') and model.data_format == 'NHWC':
        logit = logit.transpose((0, 3, 1, 2)) # NHWC to NCHW
    return logit