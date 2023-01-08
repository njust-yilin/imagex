import paddle.nn as nn
import paddle.io
import paddle
import paddle.optimizer
from pathlib import Path
import numpy as np

from deepx.utils import worker_init_fn
from deepx.utils import logger
from deepx.cvlibs import metrics


def check_logits_losses(logits_list, losses):
    len_logits = len(logits_list)
    len_losses = len(losses['types'])
    if len_logits != len_losses:
        raise RuntimeError(
            'The length of logits_list should equal to the types of loss config: {} != {}.'
            .format(len_logits, len_losses))


def loss_computation(logits_list, labels, losses):
    check_logits_losses(logits_list, losses)
    loss_list = []
    for i in range(len(logits_list)):
        logits = logits_list[i]
        loss_i = losses['types'][i]
        coef_i = losses['coef'][i]
        if loss_i.__class__.__name__ == 'MixedLoss':
            mixed_loss_list = loss_i(logits, labels)
            for mixed_loss in mixed_loss_list:
                loss_list.append(coef_i * mixed_loss)
        elif loss_i.__class__.__name__ in ("KLLoss", ):
            loss_list.append(coef_i *
                             loss_i(logits_list[0], logits_list[1].detach()))
        else:
            loss_list.append(coef_i * loss_i(logits, labels))
    return loss_list


class Trainer(object):
    def __init__(
        self,
        model:nn.Layer,
        root:str,
        train_dataset:paddle.io.Dataset,
        valid_dataset:paddle.io.Dataset,
        optimizer: paddle.optimizer.Optimizer,
        losses:dict,
        num_classes=2,
        epochs:int=10,
        batch_size:int=1,
        num_workers=0,
        log_iters:int=10,
        save_interval:int=5,
        advance_training:bool=False,
        ) -> None:
        self.advance_training = advance_training
        self.save_interval = save_interval
        self.log_iters = log_iters
        self.losses = losses
        self.optimizer = optimizer
        self.model = model
        self.root = root
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_workers = num_workers
        self.num_classes = num_classes

        self.output_path = Path(root).joinpath('output')
        self.output_path.mkdir(exist_ok=True)

        batch_sampler = paddle.io.DistributedBatchSampler(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        self.train_loader = paddle.io.DataLoader(
            train_dataset, batch_sampler=batch_sampler,
            num_workers=num_workers,
            return_list=True,
            worker_init_fn=worker_init_fn
        )

        if valid_dataset is not None:
            batch_sampler = paddle.io.DistributedBatchSampler(
                valid_dataset, batch_size=1, shuffle=False, drop_last=False)
            self.valid_loader = paddle.io.DataLoader(
                valid_dataset, batch_sampler=batch_sampler, 
                num_workers=num_workers, return_list=True
            )

    def train_epoch(self, epoch:int):
        # self.model.train()
        avg_loss = 0.0
        avg_loss_list = []
        iter = 0
        num_iters = len(self.train_dataset)
        for data in self.train_loader:
            iter += self.batch_size
            images = data['img']
            labels = data['label'].astype('int64')
            # forward
            logits_list = self.model(images)
            # calculate loss
            loss_list = loss_computation(logits_list, labels, self.losses)
            loss = sum(loss_list)
            # backward
            loss.backward()
            # optimizer step
            # if the optimizer is ReduceOnPlateau, the loss is the one which has been pass into step.
            if isinstance(self.optimizer, paddle.optimizer.lr.ReduceOnPlateau):
                self.optimizer.step(loss)
            else:
                self.optimizer.step()
            # update lr
            lr = self.optimizer.get_lr()
            # update lr
            if isinstance(self.optimizer, paddle.distributed.fleet.Fleet):
                lr_sche = self.optimizer.user_defined_optimizer._learning_rate
            else:
                lr_sche = self.optimizer._learning_rate
            if isinstance(lr_sche, paddle.optimizer.lr.LRScheduler):
                lr_sche.step()
            # clear_gradients
            self.optimizer.clear_gradients()
            self.model.clear_gradients()

            avg_loss += float(loss)
            if not avg_loss_list:
                avg_loss_list = [l.numpy() for l in loss_list]
            else:
                for i in range(len(loss_list)):
                    avg_loss_list[i] += loss_list[i].numpy()
            
            if iter % 10 == 0:
                avg_loss /= 10
                logger.info(f"[TRAIN] epoch:{epoch}/{self.epochs} iter:{iter}/{num_iters} loss:{avg_loss:.4f} lr:{lr:.6f}")

    def train(self):
        paddle.set_device('gpu')
        self.model.train()
        best_mean_iou = -1.0
        for epoch in range(self.epochs):
            self.train_epoch(epoch)
            if epoch % self.save_interval == 0 or epoch == self.epochs - 1:
                # evaluate and save the model
                paddle.save(self.model.state_dict(), self.output_path.joinpath('model.pdparams').as_posix())
                paddle.save(self.optimizer.state_dict(), self.output_path.joinpath('model.pdopt').as_posix())

                if self.valid_dataset is not None and len(self.valid_dataset) != 0:
                    mean_iou, acc, _, _, _ = self.evaluate()
                    if mean_iou > best_mean_iou:
                        best_mean_iou = mean_iou
                        paddle.save(self.model.state_dict(), self.output_path.joinpath('best.pdparams').as_posix())
                        logger.info(f"Save best model as epoch {epoch}")

    def evaluate(self):
        self.model.eval()
        total_samples = len(self.valid_dataset)
        iters = len(self.valid_loader)
        logger.info(f"Start evaluating (total_samples: {total_samples},total_iters: {iters})...")

        intersect_area_all = paddle.zeros([1], dtype='int64')
        pred_area_all = paddle.zeros([1], dtype='int64')
        label_area_all = paddle.zeros([1], dtype='int64')
        with paddle.no_grad():
            for iter, data in enumerate(self.valid_loader):
                label = data['label'].astype('int64')
                images = data['img']
                logits = self.model(images)
                logit = logits[0]
                pred = paddle.argmax(logit, axis=1, keepdim=True, dtype='int32')
                intersect_area, pred_area, label_area = metrics.calculate_area(
                    pred, label, self.num_classes
                )
                intersect_area_all += intersect_area
                pred_area_all += pred_area
                label_area_all += label_area

                from deepx.utils import color_map
                from PIL import Image
                if iter % 5 == 0:
                    image = Image.fromarray(pred[0][0].numpy().astype(np.uint8), mode='P')
                    image.putpalette(color_map.COLOR_MAP)
                    image.save(f'{iter}.png')

        metrics_input = (intersect_area_all, pred_area_all, label_area_all)
        class_iou, miou = metrics.mean_iou(*metrics_input)
        acc, class_precision, class_recall = metrics.class_measurement(*metrics_input)
        kappa = metrics.kappa(*metrics_input)
        class_dice, mdice = metrics.dice(*metrics_input)
        infor = f"[EVAL] #Images: {total_samples} mIoU: {miou:.4f} Acc: {acc:.4f} Kappa: {kappa:.4f} Dice: {mdice:.4f}"
        logger.info(infor)
        logger.info(f"[EVAL] Class IoU: {np.round(class_iou, 4)}")
        logger.info(f"[EVAL] Class Precision: {np.round(class_precision, 4)}")
        logger.info(f"[EVAL] Class Recall: {np.round(class_recall)}")
        self.model.train()
        return miou, acc, class_iou, class_precision, kappa


if __name__ == '__main__':
    from deepx.cvlibs import Config
    path = '/home/imagex/Desktop/imagex/configs/pp_liteseg_optic_disc_512x512_1k.yml'
    cfg = Config.parse_from_yaml(path)
    trainer = Trainer(
        cfg.model,
        cfg.root,
        cfg.train_dataset,
        cfg.vaild_dataset,
        cfg.optimizer,
        cfg.loss,
        cfg.num_classes,
        cfg.epochs,
    )
    trainer.train()