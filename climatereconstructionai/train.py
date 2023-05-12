import os

import torch
import torch.multiprocessing

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
import numpy as np

from . import config as cfg
from .loss import get_loss
from .metrics.get_metrics import get_metrics
from .model.net import CRAINet

from .utils.io import load_ckpt, load_model, save_ckpt
from .utils.netcdfloader import NetCDFLoader, InfiniteSampler, load_steadymask
from .utils.profiler import load_profiler
from .utils import twriter, early_stopping, evaluation


def train(arg_file=None):
    
    cfg.set_train_args(arg_file)
    
    print("* Number of GPUs: ", torch.cuda.device_count())

    torch.multiprocessing.set_sharing_strategy('file_system')

    np.random.seed(cfg.loop_random_seed)
    if cfg.cuda_random_seed is not None:
        torch.manual_seed(cfg.cuda_random_seed)

    if cfg.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

    for subdir in ("", "/images", "/ckpt"):
        outdir = cfg.snapshot_dir + subdir
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    if not os.path.exists(cfg.log_dir):
        os.makedirs(cfg.log_dir)

    writer = twriter.writer()

    writer.set_hparams(cfg.passed_args)

    if cfg.lstm_steps:
        time_steps = cfg.lstm_steps
    elif cfg.gru_steps:
        time_steps = cfg.gru_steps
    elif cfg.channel_steps:
        time_steps = cfg.channel_steps
    else:
        time_steps = 0

    # create data sets
    dataset_train = NetCDFLoader(cfg.data_root_dir, cfg.data_names, cfg.mask_dir, cfg.mask_names, 'train',
                                 cfg.data_types, time_steps,
                                 apply_transform=cfg.apply_transform,
                                 apply_img_norm=cfg.apply_img_norm,
                                 apply_img_diff=cfg.apply_img_diff)
    
    dataset_val = NetCDFLoader(cfg.data_root_dir, cfg.val_names, cfg.mask_dir, cfg.mask_names, 'val', cfg.data_types,
                               time_steps, 
                               apply_img_norm=cfg.apply_img_norm,
                               apply_img_diff=cfg.apply_img_diff)
    
    iterator_train = iter(DataLoader(dataset_train, batch_size=cfg.batch_size,
                                     sampler=InfiniteSampler(len(dataset_train)),
                                     num_workers=cfg.n_threads, multiprocessing_context='fork'))
    iterator_val = iter(DataLoader(dataset_val, batch_size=cfg.batch_size,
                                   sampler=InfiniteSampler(len(dataset_val)),
                                   num_workers=cfg.n_threads, multiprocessing_context='fork'))

    steady_mask = load_steadymask(cfg.mask_dir, cfg.steady_masks, cfg.data_types, cfg.device)

    if cfg.n_target_data == 0:
        stat_target = None
    else:
        stat_target = {"mean": dataset_train.img_mean[-cfg.n_target_data:],
                       "std": dataset_train.img_std[-cfg.n_target_data:]}

    # define network model
    if len(cfg.image_sizes) - cfg.n_target_data > 1:
        model = CRAINet(img_size=cfg.image_sizes[0],
                        enc_dec_layers=cfg.encoding_layers[0],
                        pool_layers=cfg.pooling_layers[0],
                        in_channels=2 * cfg.channel_steps + 1,
                        out_channels=cfg.out_channels,
                        fusion_img_size=cfg.image_sizes[1],
                        fusion_enc_layers=cfg.encoding_layers[1],
                        fusion_pool_layers=cfg.pooling_layers[1],
                        fusion_in_channels=(len(cfg.image_sizes) - 1 - cfg.n_target_data
                                            ) * (2 * cfg.channel_steps + 1),
                        bounds=dataset_train.bounds).to(cfg.device)
    else:
        model = CRAINet(img_size=cfg.image_sizes[0],
                        enc_dec_layers=cfg.encoding_layers[0],
                        pool_layers=cfg.pooling_layers[0],
                        in_channels=2 * cfg.channel_steps + 1,
                        out_channels=cfg.out_channels,
                        bounds=dataset_train.bounds).to(cfg.device)

    # define learning rate
    if cfg.finetune:
        lr = cfg.lr_finetune
        model.freeze_enc_bn = True
    else:
        lr = cfg.lr

    
    early_stop = early_stopping.early_stopping()

    loss_comp = get_loss.LossComputation()

    # define optimizer and loss functions
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    if cfg.lr_scheduler_patience is not None:
        lr_scheduler = ReduceLROnPlateau(optimizer, 'min', patience=cfg.lr_scheduler_patience)

    # define start point
    start_iter = 0
    if cfg.resume_iter:
        ckpt_dict = load_ckpt('{}/ckpt/{}.pth'.format(cfg.snapshot_dir, cfg.resume_iter), cfg.device)
        start_iter = load_model(ckpt_dict, model, optimizer)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Starting from iter ', start_iter)
    
    elif cfg.pretrained_model:
        ckpt_dict = load_ckpt(cfg.pretrained_model, cfg.device)
        load_model(ckpt_dict, model, optimizer)
 
    prof = load_profiler(start_iter)

    if cfg.multi_gpus:
        model = torch.nn.DataParallel(model)

    i = cfg.max_iter - (cfg.n_final_models - 1) * cfg.final_models_interval
    final_models = range(i, cfg.max_iter + 1, cfg.final_models_interval)

    savelist = []
    pbar = tqdm(range(start_iter, cfg.max_iter))
    prof.start()
    for i in pbar:

        n_iter = i + 1
        lr_val = optimizer.param_groups[0]['lr']
        pbar.set_description("lr = {:.1e}".format(lr_val))

        # train model
        model.train()
        image, mask, gt = [x.to(cfg.device) for x in next(iterator_train)]
        output = model(image, mask)

        train_loss = loss_comp.get_loss(mask, steady_mask, output, gt)

        optimizer.zero_grad()
        train_loss['total'].backward()
        optimizer.step()

        if (cfg.log_interval and n_iter % cfg.log_interval == 0):
            writer.update_scalars(train_loss, n_iter, 'train')
         
            model.eval()
            val_losses = []
            for _ in range(cfg.n_iters_val): 
                image, mask, gt = [x.to(cfg.device) for x in next(iterator_val)]
                with torch.no_grad():
                    output = model(image, mask)
                val_losses.append(list(loss_comp.get_loss(mask, steady_mask, output, gt).values()))

            val_loss = torch.tensor(val_losses).mean(dim=0)
            val_loss = dict(zip(train_loss.keys(),val_loss))

            early_stop.update(val_loss['total'].item() , n_iter, model_save=model)
            
            writer.update_scalars(val_loss, n_iter, 'val')

            if cfg.early_stopping:
                writer.update_scalar('val', 'loss_gradient', early_stop.criterion_diff , n_iter)
                                       
            if cfg.save_snapshot_image:
                fig = evaluation.create_snapshot_image(model, dataset_val, '{:s}/images/iter_{:d}'.format(cfg.snapshot_dir, n_iter))
           
            if cfg.lr_scheduler_patience is not None:
                lr_scheduler.step(val_loss['total'])
        
            
        if n_iter % cfg.save_model_interval == 0:
            save_ckpt('{:s}/ckpt/{:d}.pth'.format(cfg.snapshot_dir, n_iter), stat_target,
                      [(str(n_iter), n_iter, model, optimizer)])

        if n_iter in final_models:
            savelist.append((str(n_iter), n_iter, copy.deepcopy(model), copy.deepcopy(optimizer)))

        prof.step()

        if cfg.early_stopping and early_stop.terminate:
            prof.stop()
            model = early_stop.best_model
            break
        
    prof.stop()

    save_ckpt('{:s}/ckpt/best.pth'.format(cfg.snapshot_dir, early_stop.global_iter_best), stat_target,
                      [(str(early_stop.global_iter_best), early_stop.global_iter_best, early_stop.best_model, optimizer)])

    save_ckpt('{:s}/ckpt/final.pth'.format(cfg.snapshot_dir), stat_target, savelist)

    # run final validation over n_iters_val
    if cfg.val_metrics is not None:
        val_metrics = []
        for _ in range(cfg.n_iters_val): 
            image, mask, gt = [x.to(cfg.device) for x in next(iterator_val)]
            with torch.no_grad():
                output = model(image, mask)
            metric_dict = get_metrics(mask, steady_mask, output, gt, 'val')
            val_metrics.append(list(metric_dict.values()))
        val_metrics = torch.tensor(val_metrics).mean(dim=0)

        metric_dict = dict(zip(metric_dict.keys(),val_metrics))
        if cfg.early_stopping:
            metric_dict.update({'iterations': n_iter, 'iterations_best_model': early_stop.global_iter_best})
        writer.update_hparams(metric_dict, n_iter)

    writer.add_visualizations(mask, steady_mask, output, gt, n_iter, 'val')

    writer.close()

if __name__ == "__main__":
    train()
