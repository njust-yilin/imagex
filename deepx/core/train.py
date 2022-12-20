import paddle.nn, paddle.amp
import paddle.optimizer
from paddle.io import DataLoader, DistributedBatchSampler
import os, time
from collections import deque
import shutil

from deepx.core.val import evaluate
from deepx.datasets import Dataset
from deepx.utils import (TimeAverager, calculate_eta, resume, 
    logger, worker_init_fn, op_flops_funs, train_profiler)

# load model
# load dataset
# define optimizer and loss function
# train

def check_logits_losses(logits_list, losses):
    len_logits = len(logits_list)
    len_losses = len(losses['types'])
    if len_logits != len_losses:
        raise RuntimeError(
            'The length of logits_list should equal to the types of loss config: {} != {}.'
            .format(len_logits, len_losses))


def loss_computation(logits_list, labels, edges, losses):
    check_logits_losses(logits_list, losses)
    loss_list = []
    for i in range(len(logits_list)):
        logits = logits_list[i]
        loss_i = losses['types'][i]
        coef_i = losses['coef'][i]
        if loss_i.__class__.__name__ in ('BCELoss', ) and loss_i.edge_label:
            # Use edges as labels According to loss type.
            loss_list.append(coef_i * loss_i(logits, edges))
        elif loss_i.__class__.__name__ == 'MixedLoss':
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
        model: paddle.nn.Layer,
        train_dataset: Dataset,
        valid_dataset: Dataset,
        losses: dict,
        optimizer: paddle.optimizer.Optimizer,
        epochs: int=500, 
        batch_size: int=1,
        save_interval=50,
        log_iters=10,
        num_workers=0,
        precision='fp32',
        amp_level='O1',
        resume_model:str = None,
        use_vdl=True,
        keep_checkpoint_max=5,
        test_config=None,
        profiler_options=None):

        self._profiler_options = profiler_options
        self._train_dataset = train_dataset
        self._valid_dataset = valid_dataset
        self._losses = losses
        self._epochs = epochs
        self._batch_size = batch_size
        self._save_dir = os.path.join(train_dataset.dataset_root, 'output')
        self._save_interval = save_interval
        self._log_iters = log_iters
        self._num_workers = num_workers
        self._precision = precision
        self._amp_level = amp_level
        self._optimizer = optimizer
        self._model = model
        self._resume_model = resume_model
        self._use_vdl = use_vdl
        self._keep_checkpoint_max = keep_checkpoint_max
        self._test_config = test_config if test_config else None
        self.initialize()

    def initialize(self):
        self._nranks = paddle.distributed.ParallelEnv().nranks
        self._local_rank = paddle.distributed.ParallelEnv().local_rank

        # use amp
        if self._precision == 'fp16':
            logger.info('use AMP to train. AMP level = {self._amp_level}')
            self._scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
            if self._amp_level == 'O2':
                self._model, self._optimizer = paddle.amp.decorate(
                    models=self._model,
                    optimizers=self._optimizer,
                    level='O2',
                    save_dtype='float32')

    def train_epoch(
        self,
        model: paddle.nn.Layer,
        ddp_model: paddle.nn.Layer,
        loader: DataLoader,
        optimizer: paddle.optimizer.Optimizer,
        log_writer,
        epoch,
        iters_per_epoch
        ):
        """训练一个epoch"""

        iter = 0
        reader_cost = TimeAverager()
        batch_cost = TimeAverager()
        batch_start = time.time()

        avg_loss = 0.0
        avg_loss_list = []

        for data in loader:
            # 记录开始时间
            reader_cost.record(time.time() - batch_start)
            # 训练1个batch
            iter += 1
            # 准备数据
            images = data['img']
            labels = data['label'].astype('int64')
            edges = None
            if 'edge' in data.keys():
                edges = data['edge'].astype('int64')
            if hasattr(model, 'data_format') and model.data_format == 'NHWC':
                images = images.transpose((0, 2, 3, 1))

            if self._precision == 'fp16':
                with paddle.amp.auto_cast(
                    level=self._amp_level,
                    enable=True,
                    custom_white_list={"elementwise_add", "batch_norm", "sync_batch_norm"},
                    custom_black_list={'bilinear_interp_v2'}):

                    # forward
                    logits_list = ddp_model(images) if self._nranks > 1 else model(images)
                    # calculate loss
                    loss_list = loss_computation(
                        logits_list=logits_list,
                        labels=labels,
                        edges=edges,
                        losses=self._losses)
                    loss = sum(loss_list)

                scaled = self._scaler.scale(loss)  # scale the loss
                scaled.backward()  # do backward
                if isinstance(optimizer, paddle.distributed.fleet.Fleet):
                    self._scaler.minimize(optimizer.user_defined_optimizer, scaled)
                else:
                    self._scaler.minimize(optimizer, scaled)  # update parameters
            else:
                # forward
                logits_list = ddp_model(images) if self._nranks > 1 else model(images)
                # calculate loss
                loss_list = loss_computation(
                    logits_list=logits_list,
                    labels=labels,
                    edges=edges,
                    losses=self._losses)
                loss = sum(loss_list)

                # do backward
                loss.backward()
                # if the optimizer is ReduceOnPlateau, the loss is the one which has been pass into step.
                if isinstance(optimizer, paddle.optimizer.lr.ReduceOnPlateau):
                    optimizer.step(loss)
                else:
                    optimizer.step()

            # update lr
            lr = optimizer.get_lr()
            if isinstance(optimizer, paddle.distributed.fleet.Fleet):
                lr_sche = optimizer.user_defined_optimizer._learning_rate
            else:
                lr_sche = optimizer._learning_rate
            if isinstance(lr_sche, paddle.optimizer.lr.LRScheduler):
                lr_sche.step()

            train_profiler.add_profiler_step(self._profiler_options)

            # 清除梯度 为下一次训练作准备
            model.clear_gradients()
            avg_loss += float(loss)
            if not avg_loss_list:
                avg_loss_list = [l.numpy() for l in loss_list]
            else:
                for i in range(len(loss_list)):
                    avg_loss_list[i] += loss_list[i].numpy()
            batch_cost.record(time.time() - batch_start, num_samples=self._batch_size)

            # 打印训练结果
            if (iter) % self._log_iters == 0 and self._local_rank == 0:
                # log
                avg_loss /= self._log_iters
                remain_iters = iters_per_epoch * (self._epochs - epoch) - iter
                avg_train_batch_cost = batch_cost.get_average()
                avg_train_reader_cost = reader_cost.get_average()
                eta = calculate_eta(remain_iters, avg_train_batch_cost)

                logger.info(
                    "[TRAIN] epoch: {}/{}, iter: {}/{}, loss: {:.4f}, lr: {:.6f}, batch_cost: {:.4f}, reader_cost: {:.5f}, ips: {:.4f} samples/sec | ETA {}"
                    .format(epoch+1, self._epochs, iter, iters_per_epoch, avg_loss,
                            lr, avg_train_batch_cost, avg_train_reader_cost,
                            batch_cost.get_ips_average(), eta))
                
                if self._use_vdl:
                    current_iter = epoch * iters_per_epoch + iter
                    log_writer.add_scalar('Train/loss', avg_loss, current_iter)
                    # Record all losses if there are more than 2 losses.
                    if len(avg_loss_list) > 1:
                        avg_loss_dict = {}
                        for i, value in enumerate(avg_loss_list):
                            avg_loss_dict['loss_' + str(i)] = value
                        for key, value in avg_loss_dict.items():
                            log_tag = 'Train/' + key
                            log_writer.add_scalar(log_tag, value, current_iter)

                    log_writer.add_scalar('Train/lr', lr, current_iter)
                    log_writer.add_scalar('Train/batch_cost', avg_train_batch_cost, current_iter)
                    log_writer.add_scalar('Train/reader_cost', avg_train_reader_cost, current_iter)

            batch_start = time.time()


    def train(self):
        # 准备模型 优化器 评价函数
        model = self._model
        optimizer = self._optimizer

        # 恢复训练
        start_epoch = 0
        if self._resume_model is not None:
            start_epoch = resume(model, optimizer, self._resume_model)

        # 创建输出目录
        os.system(f'rm -rf {self._save_dir}')
        os.makedirs(self._save_dir, exist_ok=True)

        # 多卡操作
        ddp_model = None
        if self._nranks > 1:
            paddle.distributed.fleet.init(is_collective=True)
            optimizer = paddle.distributed.fleet.distributed_optimizer(optimizer)  
            # The return is Fleet object
            ddp_model = paddle.distributed.fleet.distributed_model(model)

        # 准备训练集
        batch_sampler = DistributedBatchSampler(
            self._train_dataset, 
            batch_size=self._batch_size, 
            shuffle=True, 
            drop_last=True)
        loader = DataLoader(
            self._train_dataset,
            batch_sampler=batch_sampler,
            num_workers=self._num_workers,
            return_list=True,
            worker_init_fn=worker_init_fn)

        # 初始化vdl
        log_writer = None
        if self._use_vdl:
            from visualdl import LogWriter
            log_writer = LogWriter(self._save_dir)
        
        # 初始化参数
        iters_per_epoch = len(batch_sampler)
        best_mean_iou = -1.0
        best_model_iter = -1
        save_models = deque()

        for epoch in range(start_epoch, self._epochs):
            model.train()
            self.train_epoch(
                model=model,
                ddp_model=ddp_model,
                loader=loader,
                optimizer=optimizer,
                log_writer=log_writer,
                epoch=epoch,
                iters_per_epoch=iters_per_epoch
            )
        
            num_workers = 1 if self._num_workers > 0 else 0

            test_config = {}
            mean_iou, acc, _, _, _ = evaluate(
                model,
                self._valid_dataset,
                num_workers=num_workers,
                precision=self._precision,
                amp_level=self._amp_level,
                **test_config)

            if (epoch % self._save_interval == 0 or epoch + 1 == self._epochs) and self._local_rank == 0:
                current_save_dir = os.path.join(self._save_dir, f"epoch_{epoch}")
                if not os.path.isdir(current_save_dir):
                    os.makedirs(current_save_dir)
                paddle.save(model.state_dict(),
                            os.path.join(current_save_dir, 'model.pdparams'))
                paddle.save(optimizer.state_dict(),
                            os.path.join(current_save_dir, 'model.pdopt'))
                save_models.append(current_save_dir)
                if len(save_models) > self._keep_checkpoint_max > 0:
                    model_to_remove = save_models.popleft()
                    shutil.rmtree(model_to_remove)

                if self._valid_dataset is not None:
                    if mean_iou > best_mean_iou:
                        best_mean_iou = mean_iou
                        best_model_iter = iter
                        best_model_dir = os.path.join(self._save_dir, "best_model")
                        paddle.save(
                            model.state_dict(),
                            os.path.join(best_model_dir, 'model.pdparams'))
                        paddle.save(optimizer.state_dict(),
                            os.path.join(best_model_dir, 'model.pdopt'))
                    logger.info(
                        '[EVAL] The model with the best validation mIoU ({:.4f}) was saved at iter {}.'
                        .format(best_mean_iou, best_model_iter))

                    if self._use_vdl:
                        log_writer.add_scalar('Evaluate/mIoU', mean_iou, epoch)
                        log_writer.add_scalar('Evaluate/Acc', acc, epoch)
            
        # Calculate flops.
        if self._local_rank == 0 and not (self._precision == 'fp16' and self._amp_level == 'O2'):
            for data in loader:
                images = data['img']
                break
            _, c, h, w = images.shape
            _ = paddle.flops(
                model, [1, c, h, w],
                custom_ops={paddle.nn.SyncBatchNorm: op_flops_funs.count_syncbn})

        # Sleep for a second to let dataloader release resources.
        time.sleep(1)
        if self._use_vdl:
            log_writer.close()

    def valid(self, dataset: Dataset):
        pass


if __name__ == '__main__':
    from deepx.cvlibs import Config
    path = '/home/imagex/Desktop/yilin/imagex/notbook/pp_liteseg_optic_disc_512x512_1k.yml'
    config = Config.parse_from_yaml(path)
    trainer = Trainer(
        config.model,
        config.train_dataset,
        config.valid_dataset,
        config.loss,
        config.optimizer,
        epochs=config.epoch,
        save_interval=1
        # resume_model='/home/imagex/imagex_data/networks/optic_disc_seg/output/epoch_9'
    )
    trainer.train()