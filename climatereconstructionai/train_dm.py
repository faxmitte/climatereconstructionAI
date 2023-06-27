import os

import torch
import torch.multiprocessing

from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
import numpy as np

from . import config as cfg
from .metrics.get_metrics import get_metrics
from .model.net import CRAINet
from .model import DM as DM
from .model import diffusion as diffusion

from .utils.io import load_ckpt, load_model, save_ckpt
from .utils.netcdfloader import NetCDFLoader, InfiniteSampler, load_steadymask, img_norm
from .utils.profiler import load_profiler
from .utils import twriter, early_stopping, evaluation
from .model import unet as unet


def train_dm(arg_file=None):
   
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
                                 apply_img_norm=cfg.apply_img_norm)
    dataset_val = NetCDFLoader(cfg.data_root_dir, cfg.val_names, cfg.mask_dir, cfg.mask_names, 'val', cfg.data_types,
                               time_steps,
                               apply_img_norm=cfg.apply_img_norm)
    iterator_train = iter(DataLoader(dataset_train, batch_size=cfg.batch_size,
                                     sampler=InfiniteSampler(len(dataset_train)),
                                     num_workers=cfg.n_threads))
    iterator_val = iter(DataLoader(dataset_val, batch_size=cfg.batch_size,
                                   sampler=InfiniteSampler(len(dataset_val)),
                                   num_workers=cfg.n_threads))

    steady_mask = load_steadymask(cfg.mask_dir, cfg.steady_masks, cfg.data_types, cfg.device)

    if cfg.n_target_data == 0:
        stat_target = None
    else:
        stat_target = {"mean": dataset_train.img_mean[-cfg.n_target_data:],
                       "std": dataset_train.img_std[-cfg.n_target_data:]}
        
    diffusion_settings = cfg.diffusion_settings
    if 'use_crai' in diffusion_settings.keys() and diffusion_settings['use_crai']:
        use_crai = True
    else: 
        use_crai = False

    if use_crai:
        if len(cfg.image_sizes) - cfg.n_target_data > 1:
            model = CRAINet(img_size_source=dataset_train[0][0].shape[-2:],
                            img_size_target=dataset_train[0][-1].shape[-2:],
                            enc_dec_layers=cfg.encoding_layers[0],
                            pool_layers=cfg.pooling_layers[0],
                            in_channels=2,
                            out_channels=cfg.out_channels,
                            fusion_img_size=cfg.image_sizes[1],
                            fusion_enc_layers=cfg.encoding_layers[1],
                            fusion_pool_layers=cfg.pooling_layers[1],
                            fusion_in_channels=(len(cfg.image_sizes) - 1 - cfg.n_target_data
                                                ) * (2 * cfg.channel_steps + 1),
                            bounds=None).to(cfg.device)
        else:
            model = CRAINet(img_size_source=dataset_train[0][0].shape[-2:],
                            img_size_target=dataset_train[0][-1].shape[-2:],
                            enc_dec_layers=cfg.encoding_layers[0],
                            pool_layers=cfg.pooling_layers[0],
                            in_channels=2,
                            out_channels=cfg.out_channels,
                            bounds=None).to(cfg.device)

    

    if 'use_custom_dm' in diffusion_settings.keys() and diffusion_settings['use_custom_dm']:
        use_custom_dm = True
    else: 
        use_custom_dm = False

    schedule_opt = diffusion_settings['schedule_opt']

    if not use_crai:
        model_opt = diffusion_settings['model_opt']
        model = unet.UNet(
            in_channel=model_opt['in_channel'],
            attn_res=model_opt['attn_res'],
            res_blocks=model_opt['res_blocks'],
            out_channel=model_opt['out_channel'],
            inner_channel=model_opt['inner_channel'],
            dropout=model_opt['dropout'], 
            image_size=cfg.image_sizes[0]
            ).to(cfg.device)

    if not use_custom_dm:
        dm_model = diffusion.GaussianDiffusion(
            model, 
            img_size_source=dataset_train[0][0].shape[-1],
            img_size_target=dataset_train[0][-1].shape[-1],
            channels=1, 
            use_crai=diffusion_settings['use_crai']
            )
        dm_model.set_new_noise_schedule(schedule_opt, cfg.device)
        dm_model.set_loss(cfg.device)
    else:
        dm_model = DM.DM(
            schedule_opt['n_timestep'], 
            gmodel=None, 
            min_val=schedule_opt['linear_start'], 
            max_val=schedule_opt['linear_end'], 
            use_sigma=False, 
            use_mu=False
            ).to(cfg.device)

    # define learning rate
    if cfg.finetune:
        lr = cfg.lr_finetune
        model.freeze_enc_bn = True
    else:
        lr = cfg.lr
    
    early_stop = early_stopping.early_stopping()

    # define optimizer and loss functions
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # define start point
    start_iter = 0
    
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

        if use_custom_dm:
            train_loss, _, _ = dm_model.diffusion_loss(model,[image, mask, gt])

        else:
            
            x_in = {'SR':image[:,0,:,:,:], 'HR':gt[:,0,:,:,:], 'mask': mask[:,0,:,:,:]}
                     
            loss = dm_model(x_in)

            b, c, h, w = x_in['HR'].shape
            loss = loss.sum()/int(b*c*h*w)
            train_loss = {'total': loss}

        optimizer.zero_grad()
        train_loss['total'].backward()
        optimizer.step()
        
        early_stop.update(train_loss['total'].item() , n_iter, model_save=model)

        if (cfg.log_interval and n_iter % cfg.log_interval == 0):
            writer.update_scalars(train_loss, n_iter, 'train')
                                       
            
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

    writer.close()

if __name__ == "__main__":
    train_dm()
