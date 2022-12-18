import paddle


class EMA(object):
    def __init__(self, model:paddle.nn.Layer, decay=0.99) -> None:
        assert decay >= 0 and decay <= 1.0, \
            f"The decay = {decay} should in [0.0, 1.0]"
        
        self._model = model
        self._decay = decay
        self._ema_data = {}
        self._backup_data = {}

        for name, param in self._model.named_parameters():
            if not param.stop_gradient:
                self._ema_data[name] = param.numpy()
    
    def aplay(self):
        for name, param in self._model.named_parameters():
            if not param.stop_gradient:
                self._backup_data[name] = param.numpy()
                param.set_value(self._ema_data[name])
    
    def restore(self):
        for name, param in self._model.named_parameters():
            if not param.stop_gradient:
                param.set_value(self._backup_data[name])
        self._backup_data = {}

    def step(self):
        for name, param in self._model.named_parameters():
            if not param.stop_gradient:
                self._ema_data[name] = self._decay * self._ema_data[name] \
                    + (1.0 - self._decay) * param.numpy()


if __name__ == '__main__':
    model = paddle.paddle.vision.mobilenet_v1()
    ema = EMA(model)
    ema.step()
    ema.aplay()
    ema.restore()