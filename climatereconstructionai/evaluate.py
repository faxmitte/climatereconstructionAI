import os

import h5py
from numpy import ma
import numpy as np
from .model.net import CRAINet
from .utils.evaluation import infill, create_evaluation_report, create_evaluation_graphs, create_evaluation_maps, \
    create_evaluation_images, create_scatter_plots
from .utils.io import load_ckpt
from . import config as cfg


def evaluate(arg_file=None, prog_func=None):
    cfg.set_evaluate_args(arg_file, prog_func)

    if not os.path.exists(cfg.log_dir):
        os.makedirs(cfg.log_dir)

    n_models = len(cfg.model_names)
    assert n_models == len(cfg.eval_names)

    if cfg.use_etienne_ncloader:
        from .utils.netcdfloader import EtienneNetCDFLoader as NetCDFLoader
    else:
        from .utils.netcdfloader import JohannesNetCDFLoader as NetCDFLoader

    if cfg.infill:
        for i_model in range(n_models):

            if cfg.lstm_steps:
                time_steps = cfg.lstm_steps
            elif cfg.gru_steps:
                time_steps = cfg.gru_steps
            elif cfg.channel_steps:
                time_steps = cfg.channel_steps
            else:
                time_steps = 0

            dataset_val = NetCDFLoader(cfg.data_root_dir, cfg.img_names, cfg.mask_dir, cfg.mask_names, "infill",
                                       cfg.data_types, time_steps)

            if len(cfg.image_sizes) > 1:
                model = CRAINet(img_size=cfg.image_sizes[0],
                                enc_dec_layers=cfg.encoding_layers[0],
                                pool_layers=cfg.pooling_layers[0],
                                in_channels=2 * cfg.channel_steps + 1,
                                out_channels=cfg.out_channels,
                                fusion_img_size=cfg.image_sizes[1],
                                fusion_enc_layers=cfg.encoding_layers[1],
                                fusion_pool_layers=cfg.pooling_layers[1],
                                fusion_in_channels=(len(cfg.image_sizes) - 1) * (2 * cfg.channel_steps + 1)).to(cfg.device)
            else:
                model = CRAINet(img_size=cfg.image_sizes[0],
                                enc_dec_layers=cfg.encoding_layers[0],
                                pool_layers=cfg.pooling_layers[0],
                                in_channels=2 * cfg.channel_steps + 1,
                                out_channels=cfg.out_channels).to(cfg.device)

            load_ckpt("{}/{}".format(cfg.model_dir, cfg.model_names[i_model]), [('model', model)], cfg.device)

            model.eval()

            infill(model, dataset_val, "{}/{}".format(cfg.evaluation_dirs[0], cfg.eval_names[i_model]))

    gt, outputs = None, None

    # create report
    if cfg.create_report:
        gt, outputs = load_data(gt, outputs)
        create_evaluation_report(gt, outputs)

    # create graphs
    if cfg.create_graphs:
        if not os.path.exists('{}/graphs'.format(cfg.evaluation_dirs[0])):
            os.makedirs('{}/graphs'.format(cfg.evaluation_dirs[0]))
        gt, outputs = load_data(gt, outputs)
        create_evaluation_graphs(gt, outputs)

    # create maps
    if cfg.create_rmse_maps or cfg.create_timcor_maps or cfg.create_sum_maps:
        if not os.path.exists('{}/maps'.format(cfg.evaluation_dirs[0])):
            os.makedirs('{}/maps'.format(cfg.evaluation_dirs[0]))
        gt, outputs = load_data(gt, outputs)
        create_evaluation_maps(gt, outputs)

    # create scatter plots
    if cfg.create_scatter:
        if not os.path.exists('{}/scatter'.format(cfg.evaluation_dirs[0])):
            os.makedirs('{}/scatter'.format(cfg.evaluation_dirs[0]))
        gt, outputs = load_data(gt, outputs)
        create_scatter_plots(gt, outputs)

    # create images
    if cfg.create_images:
        if not os.path.exists('{}/images'.format(cfg.evaluation_dirs[0])):
            os.makedirs('{}/images'.format(cfg.evaluation_dirs[0]))
        r = (int(cfg.create_images[0]), int(cfg.create_images[1]))
        gt = h5py.File('{}/{}{}'.format(cfg.evaluation_dirs[0], cfg.eval_names[0], '_gt.nc'), 'r').get(cfg.data_types[0])[r[0]:r[1], :, :]
        mask = h5py.File('{}/{}{}'.format(cfg.evaluation_dirs[0], cfg.eval_names[0], '_mask.nc'), 'r').get(cfg.data_types[0])[r[0]:r[1], :, :]
        image = h5py.File('{}/{}{}'.format(cfg.evaluation_dirs[0], cfg.eval_names[0], '_image.nc'), 'r').get(cfg.data_types[0])[r[0]:r[1], :,
                :]
        if gt.ndim == 4:
            gt = gt[:, 0, :, :]
        if mask.ndim == 4:
            mask = mask[:, 0, :, :]
        if image.ndim == 4:
            image = image[:, 0, :, :]
        image = ma.masked_array(image, 1 - mask)[:, :, :]

        data_sets = {'GT': gt, 'Input': image}

        for i in range(len(cfg.evaluation_dirs)):
            output = h5py.File('{}/{}{}'.format(cfg.evaluation_dirs[i], cfg.eval_names[0], '_infilled.nc'), 'r').get(cfg.data_types[0])[
                     r[0]:r[1],
                     :, :]
            if output.ndim == 4:
                output = output[:, 0, :, :]
            output[output < 0.0] = 0.0
            data_sets[cfg.infilled_names[i]] = output

        create_video = False
        if cfg.create_video:
            create_video = True
        for key, value in data_sets.items():
            create_evaluation_images(key, value, create_video)


# loads data from infilled data sets
def load_data(gt, outputs):
    if gt is not None and outputs is not None:
        return gt, outputs

    if cfg.eval_range:
        r = (int(cfg.eval_range[0]), int(cfg.eval_range[1]))
        gt = h5py.File('{}/{}{}'.format(cfg.evaluation_dirs[0], cfg.eval_names[0], '_gt.nc'), 'r').get(cfg.data_types[0])[r[0]:r[1], :, :]
        mask = h5py.File('{}/{}{}'.format(cfg.evaluation_dirs[0], cfg.eval_names[0], '_mask.nc'), 'r').get(cfg.data_types[0])
        if len(mask) > 1:
            mask = mask[r[0]:r[1], :, :]
    else:
        gt = h5py.File('{}/{}{}'.format(cfg.evaluation_dirs[0], cfg.eval_names[0], '_gt.nc'), 'r').get(cfg.data_types[0])[:, :, :]
        mask = h5py.File('{}/{}{}'.format(cfg.evaluation_dirs[0], cfg.eval_names[0], '_mask.nc'), 'r').get(cfg.data_types[0])[:, :, :]
    if gt.ndim == 4:
        gt = gt[:, 0, :, :]
    if mask.ndim == 4:
        mask = mask[:, 0, :, :]
    if len(gt) != len(mask):
        mask = np.repeat(mask, len(gt), axis=0)
    if cfg.eval_threshold:
        mask[gt < cfg.eval_threshold] = 1
    nan_mask = np.isnan(gt)
    combined_mask = np.logical_or(mask, nan_mask)
    gt = np.ma.array(gt, mask=combined_mask)
    outputs = {}
    for i in range(len(cfg.evaluation_dirs)):
        if cfg.eval_range:
            r = (int(cfg.eval_range[0]), int(cfg.eval_range[1]))
            output = h5py.File('{}/{}{}'.format(cfg.evaluation_dirs[i], cfg.eval_names[0], '_infilled.nc'), 'r').get(cfg.data_types[0])[
                     r[0]:r[1], :, :]
        else:
            output = h5py.File('{}/{}{}'.format(cfg.evaluation_dirs[i], cfg.eval_names[0], '_infilled.nc'), 'r').get(cfg.data_types[0])[:, :,
                     :]
        if output.ndim == 4:
            output = output[:, 0, :, :]
        output[output < 0.0] = 0.0
        output = np.ma.array(output, mask=combined_mask)

        outputs[cfg.infilled_names[i]] = output
    return gt, outputs


if __name__ == "__main__":
    evaluate()
