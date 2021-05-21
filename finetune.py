import os, sys
import math

import hydra
import torch
import timm
from hydra.utils import instantiate
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import NativeScaler

import models
from data import create_dataloader
from utils import MetricLogger, SmoothedValue
from utils import fix_random_seed



@hydra.main(config_path='./configs', config_name='finetune')
def main(cfg):
    if cfg.seed is not None:
        fix_random_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True

    # dataloader
    trainloader, num_classes = create_dataloader(cfg.data)

    # additional data augmentation (mixup/cutmix)
    mixup_fn = None
    mixup_enable = (cfg.data.mixup.mixup_alpha > 0.) or (cfg.data.mixup.cutmix_alpha > 0.)
    if mixup_enable:
        mixup_fn = instantiate(cfg.data.mixup, num_classes=num_classes)
        print(f'MixUp/Cutmix was enabled\n')

    # create model
    model = instantiate(cfg.model, num_classes=num_classes)
    print(f'Model[{cfg.model.model_name}] was created')
    
    # load pretrained weights
    if cfg.ckpt is not None:
        ckpt = torch.load(cfg.ckpt, map_location='cpu')
        ckpt_model = ckpt['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias', 'fc.weight', 'fc.bias']:
            if k in ckpt_model and ckpt_model[k].shape != state_dict[k].shape:
                print(f'Remove key [{k}] from pretrained checkpoint')
                del ckpt_model[k]

        model.load_state_dict(ckpt_model, strict=False)
        print(f'Checkpoint was loaded from {cfg.ckpt}\n')
    else:
        print(f'Model[{cfg.model.model_name}] will be trained from scratch') 
    
    # wrap model with DP
    model = torch.nn.parallel.DataParallel(model)
    model.cuda()
    model_without_dp = model.module

    # optimizer
    scaled_lr = cfg.optim.args.lr * cfg.data.loader.batch_size / 512.0
    cfg.optim.args.lr = scaled_lr
    optimizer = instantiate(cfg.optim, model=model)
    print(f'Optimizer: \n{optimizer}\n')

    # scheduler
    lr_scheduler, _ = instantiate(cfg.scheduler, optimizer=optimizer)
    print(f'Scheduler: \n{lr_scheduler}\n')
   
    # criterion
    if cfg.data.mixup.mixup_alpha > 0.:
        criterion = SoftTargetCrossEntropy().cuda()
        print('SoftTargetCrossEntropy is used for criterion\n')
    elif cfg.data.mixup.label_smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(cfg.data.mixup.label_smoothing).cuda()
        print('LabelSmoothingCrossEntropy is used for criterion\n')
    else:
        criterion = torch.nn.CrossEntropyLoss().cuda()
        print('CrossEntropyLoss is used for criterion\n')
    loss_scaler = NativeScaler()

    # load resume
    start_epoch = 1
    if cfg.resume is not None:
        checkpoint = torch.load(cfg.resume, map_location='cpu')
        model_without_dp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        loss_scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
        print(f'Resume was loaded from {cfg.resume}\n')

    print(f'Start training for {cfg.epochs} epochs')
    for epoch in range(start_epoch, cfg.epochs + 1):
        # train one epoch
        model.eval()
        metric_logger = MetricLogger(delimiter=' ')
        metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = f'Epoch: [{epoch:03}/{cfg.epochs:03}]'
        for data in metric_logger.log_every(trainloader, cfg.print_iter_freq, header):
            images = data[0].cuda(non_blocking=True)
            labels = data[1].cuda(non_blocking=True)

            if mixup_fn is not None:
                images, labels = mixup_fn(images, labels)
            
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print(f'Loss is {loss_value}, stopping training')
                sys.exit(1)

            optimizer.zero_grad()
            is_second_order = (hasattr(optimizer, 'is_second_order')) and (optimizer.is_second_order)
            loss_scaler(
                loss=loss,
                optimizer=optimizer,
                parameters=model.parameters(),
                create_graph=is_second_order
            )

            torch.cuda.synchronize()
            
            metric_logger.update(loss=loss_value)
            metric_logger.update(lr=optimizer.param_groups[0]['lr'])
        
        # gather the stats from all process
        metric_logger.synchronize_between_processes()
        print(f'Averaged stats: {metric_logger}')

        lr_scheduler.step(epoch)

        if epoch % cfg.save_epoch_freq == 0:
            save_path = f'{os.getcwd()}/{cfg.model.model_name}_{cfg.data.name}_{epoch:03}ep.pth'
            torch.save({
                'model': model_without_dp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'scaler': loss_scaler.state_dict(),
                'epoch': epoch
            }, save_path)

    save_path = f'{os.getcwd()}/{cfg.model.model_name}_{cfg.data.name}_{epoch:03}ep.pth'
    torch.save({
        'model': model_without_dp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'scaler': loss_scaler.state_dict(),
        'epoch': epoch
    }, save_path)

if __name__ == '__main__':
    main()
