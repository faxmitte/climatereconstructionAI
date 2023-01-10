import os.path
from tensorboardX import SummaryWriter

from .netcdfchecker import reformat_dataset
from .netcdfloader import load_steadymask
from .normalizer import renormalize
from .plotdata import plot_data
from .. import config as cfg
import torch
import netCDF4
import numpy as np
import pandas as pd
import imageio
import matplotlib.pyplot as plt
import calendar
import sys, os

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import font_manager
from netCDF4 import Dataset
from fpdf import FPDF

from . import metrics

def create_snapshot_image(model, dataset, filename):
    image, mask, gt, fusion_image, fusion_mask, fusion_gt = zip(*[dataset[int(i)] for i in cfg.eval_timesteps])

    image = torch.stack(image).to(cfg.device)
    mask = torch.stack(mask).to(cfg.device)
    gt = torch.stack(gt).to(cfg.device)
    if fusion_image:
        fusion_image = torch.stack(fusion_image).to(cfg.device)
        fusion_mask = torch.stack(fusion_mask).to(cfg.device)

    with torch.no_grad():
        output = model(image, mask, fusion_image, fusion_mask)

    # select last element of lstm sequence as evaluation element
    image = image[:, cfg.recurrent_steps, cfg.gt_channels, :, :].to(torch.device('cpu'))
    gt = gt[:, cfg.recurrent_steps, cfg.gt_channels, :, :].to(torch.device('cpu'))
    mask = mask[:, cfg.recurrent_steps, cfg.gt_channels, :, :].to(torch.device('cpu'))
    output = output[:, cfg.recurrent_steps, :, :, :].to(torch.device('cpu'))

    output_comp = mask * image + (1 - mask) * output

    # set mask
    mask = 1 - mask
    image = np.ma.masked_array(image, mask)
    mask = np.ma.masked_array(mask, mask)

    for c in range(output.shape[1]):

        if cfg.vlim is None:
            vmin = gt[:, c, :, :].min().item()
            vmax = gt[:, c, :, :].max().item()
        else:
            vmin = cfg.vlim[0]
            vmax = cfg.vlim[1]
        data_list = [image[:, c, :, :], mask[:, c, :, :], output[:, c, :, :], output_comp[:, c, :, :], gt[:, c, :, :]]

        # plot and save data
        fig, axes = plt.subplots(nrows=len(data_list), ncols=image.shape[0], figsize=(20, 20))
        fig.patch.set_facecolor('black')
        for i in range(len(data_list)):
            for j in range(image.shape[0]):
                axes[i, j].axis("off")
                axes[i, j].imshow(np.squeeze(data_list[i][j]), vmin=vmin, vmax=vmax)
        plt.subplots_adjust(wspace=0.012, hspace=0.012)
        plt.savefig(filename + '_' + str(c) + '.jpg', bbox_inches='tight', pad_inches=0)
    plt.clf()
    plt.close('all')


def get_partitions(parameters, length):
    if cfg.maxmem is None:
        partitions = cfg.partitions
    else:
        model_size = 0
        for parameter in parameters:
            model_size += sys.getsizeof(parameter.storage())
        model_size = model_size * length / 1e6
        partitions = int(np.ceil(model_size * 5 / cfg.maxmem))

    if partitions > length:
        partitions = length

    return partitions


def infill(model, dataset, eval_path):
    if not os.path.exists(cfg.evaluation_dirs[0]):
        os.makedirs('{:s}'.format(cfg.evaluation_dirs[0]))
    image = []
    mask = []
    gt = []
    output = []
    output_comp = []

    partitions = get_partitions(model.parameters(), dataset.img_length)

    if partitions != 1:
        print("The data will be split in {} partitions...".format(partitions))

    n_elements = dataset.__len__() // partitions
    for split in range(partitions):
        data_part = []
        i_start = split * n_elements
        if split == partitions - 1:
            i_end = dataset.__len__()
        else:
            i_end = i_start + n_elements
        for i in range(6):
            data_part.append(torch.stack([dataset[j][i] for j in range(i_start, i_end)]))

        # Tensors in data_part: image_part, mask_part, gt_part, fusion_image_part, fusion_mask_part, fusion_gt_part

        if split == 0 and cfg.create_graph:
            writer = SummaryWriter(log_dir=cfg.log_dir)
            writer.add_graph(model, [data_part[0], data_part[1], data_part[3], data_part[4]])
            writer.close()

        # get results from trained network
        with torch.no_grad():
            output_part = model(data_part[0].to(cfg.device), data_part[1].to(cfg.device),
                                data_part[3].to(cfg.device), data_part[4].to(cfg.device))

        # image_part, mask_part, gt_part
        for i in range(3):
            data_part[i] = data_part[i][:, cfg.recurrent_steps, :, :, :].to(torch.device('cpu'))
            # only select first channel
            data_part[i] = torch.unsqueeze(data_part[i][:, cfg.channel_steps, :, :], dim=1)

        output_part = output_part[:, cfg.recurrent_steps, :, :, :].to(torch.device('cpu'))

        #image.append(data_part[0])
        #mask.append(data_part[1])
        #gt.append(data_part[2])
        #output.append(output_part)
        output_comp.append(data_part[1] * data_part[0] + (1 - data_part[1]) * output_part)

    #image = torch.cat(image)
    #mask = torch.cat(mask)
    #gt = torch.cat(gt)
    #output = torch.cat(output)
    output_comp = torch.cat(output_comp)

    #steady_mask = load_steadymask(cfg.mask_dir, cfg.steady_mask, cfg.data_types[0], cfg.device)
    #if steady_mask is not None:
    #    steady_mask = 1 - steady_mask
    #    image /= steady_mask
    #    gt /= steady_mask
    #    output /= steady_mask

    # create output_comp
    #output_comp = mask * image + (1 - mask) * output
    #image[np.where(mask == 0)] = np.nan

    cvar = {'infilled': output_comp} #, 'output': output}#{'gt': gt, 'mask': mask, 'image': image, 'output': output, 'infilled': output_comp}
    create_outputs(cvar, dataset, 0, eval_path)


def create_outputs(cvar, dataset, ind_data, eval_path):
    data_type = cfg.data_types[ind_data]

    for cname in cvar:
        output_name = '{}_{}'.format(eval_path, cname)

        ds = dataset.xr_dss[1].copy()

        if cfg.normalize_data:
            cvar[cname] = renormalize(cvar[cname], dataset.img_mean[ind_data], dataset.img_std[ind_data])
        ds[data_type].values = cvar[cname].to(torch.device('cpu')).detach().numpy()[:, 0, :, :]

        ds = reformat_dataset(dataset.xr_dss[0], ds, data_type)
        ds.attrs["history"] = "Infilled using CRAI " \
                              "(Climate Reconstruction AI: https://github.com/FREVA-CLINT/climatereconstructionAI)\n" \
                              + ds.attrs["history"]
        ds.to_netcdf(output_name + ".nc")

    output_name = '{}_{}'.format(eval_path, "combined")
    plot_data(dataset.xr_dss[1].coords, [cvar["image"], cvar["infilled"]], ["Original", "Reconstructed"], output_name,
              data_type, cfg.plot_results, *cfg.dataset_format["scale"])


def create_evaluation_images(name, data_set, create_video=False, vmin=0, vmax=5, axis='off'):
    for i in range(data_set.shape[0]):
        plt.imshow(np.squeeze(data_set[i, :, :]), vmin=vmin, vmax=vmax)
        plt.axis(axis)
        plt.savefig('{}/images/{}{}.jpg'.format(cfg.evaluation_dirs[0], name, str(i)), bbox_inches='tight',
                    pad_inches=0)
        plt.clf()

    if create_video:
        with imageio.get_writer('{}/images/{}.gif'.format(cfg.evaluation_dirs[0], name), mode='I', fps=cfg.fps) as writer:
            for i in range(data_set.shape[0]):
                image = imageio.imread('{}/images/{}{}.jpg'.format(cfg.evaluation_dirs[0], name, str(i)))
                writer.append_data(image)


def create_evaluation_report(gt, outputs):
    # define arrays for dataframe
    data_sets = ['GT']
    rmses = ['0.0']
    rmses_over_mean = ['0.0']
    time_cors = ['1.0']
    total_prs = [int(metrics.total_sum(gt))]
    mean_fld_cors = ['1.0']
    fld_cor_total_sum = ['1.0']

    # define output metrics
    for output_name, output in outputs.items():
        # append values
        data_sets.append(output_name)
        rmses.append('%.5f' % metrics.rmse(gt, output))
        rmses_over_mean.append('%.5f' % metrics.rmse_over_mean(gt, output))
        time_cors.append('%.5f' % metrics.timcor(gt, output))
        total_prs.append(int(metrics.total_sum(output)))
        mean_fld_cors.append('%.5f' % metrics.timmean_fldor(gt, output))
        fld_cor_total_sum.append('%.5f' % metrics.fldor_timsum(gt, output))

    # create dataframe for metrics
    df = pd.DataFrame()
    df['Data Set'] = data_sets
    df['RMSE'] = rmses
    df['RMSE over mean'] = rmses_over_mean
    df['Time Correlation'] = time_cors
    df['Total Precipitation'] = total_prs
    df['Mean Field Correlation'] = mean_fld_cors
    df['Field Correlation of total Field Sum'] = fld_cor_total_sum

    # Create PDF plot
    labels = []
    data = []
    for output_name, output in outputs.items():
        labels.append(output_name)
        data.append(np.sum(output, axis=(1, 2)) / 3600)
    labels.append('GT')
    data.append(np.sum(gt, axis=(1, 2)) / 3600)
    plt.hist(data, bins=cfg.PDF_BINS, label=labels, edgecolor='black')
    plt.title('Probabilistic density Histogram')
    plt.xlabel('Total precipitation fall')
    plt.ylabel('Number of hours')
    plt.legend()
    plt.xscale("log")

    plt.savefig(cfg.evaluation_dirs[0] + 'pdf.png', dpi=300)
    plt.clf()

    # create PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_xy(0, 0)
    pdf.set_font('arial', 'B', 16)
    pdf.cell(60, 40)
    pdf.cell(75, 30, "Statistical evaluation metrics", 0, 2, 'C')
    pdf.cell(90, 5, " ", 0, 2, 'C')
    pdf.cell(-53)
    pdf.set_font('arial', 'B', 12)
    pdf.cell(25, 10, 'Data Set', 1, 0, 'C')
    pdf.cell(25, 10, 'RMSE', 1, 0, 'C')
    pdf.cell(35, 10, 'RMSE ov. mean', 1, 0, 'C')
    pdf.cell(25, 10, 'Time Cor', 1, 0, 'C')
    pdf.cell(25, 10, 'Total PR', 1, 0, 'C')
    pdf.cell(30, 10, 'Mean Fld Cor', 1, 0, 'C')
    pdf.cell(30, 10, 'Fld Cor Sum', 1, 2, 'C')
    pdf.cell(-165)
    pdf.set_font('arial', '', 12)
    for i in range(0, len(df)):
        pdf.cell(25, 10, '%s' % (df['Data Set'].iloc[i]), 1, 0, 'C')
        pdf.cell(25, 10, '%s' % (str(df['RMSE'].iloc[i])), 1, 0, 'C')
        pdf.cell(35, 10, '%s' % (str(df['RMSE over mean'].iloc[i])), 1, 0, 'C')
        pdf.cell(25, 10, '%s' % (str(df['Time Correlation'].iloc[i])), 1, 0, 'C')
        pdf.cell(25, 10, '%s' % (str(df['Total Precipitation'].iloc[i])), 1, 0, 'C')
        pdf.cell(30, 10, '%s' % (str(df['Mean Field Correlation'].iloc[i])), 1, 0, 'C')
        pdf.cell(30, 10, '%s' % (str(df['Field Correlation of total Field Sum'].iloc[i])), 1, 2, 'C')
        pdf.cell(-165)
    pdf.cell(-20)
    pdf.cell(130, 10, " ", 0, 2, 'C')

    pdf.add_page()

    pdf.set_font('arial', 'B', 16)
    pdf.cell(50)
    pdf.cell(75, 30, "Probabilistic Density Function", 0, 2, 'C')
    pdf.cell(90, 5, " ", 0, 2, 'C')
    pdf.cell(-60)
    pdf.image(cfg.evaluation_dirs[0] + 'pdf.png', x=None, y=None, w=208, h=218, type='', link='')

    report_name = ''
    for name in cfg.infilled_names:
        report_name += name
    pdf.output('{}/{}_{}.pdf'.format(cfg.evaluation_dirs[0], cfg.eval_names[0], report_name), 'F')


def init_font():
    font_dirs = ['../fonts/']
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)


def plot_ts(title, file_name, time_series_dict, time, unit):
    init_font()
    plt.rcParams.update({'font.family': 'Times New Roman'})
    index = 0
    for name, time_series in time_series_dict.items():
        if name == 'Ground Truth':
            param = 'k:'
        else:
            param = '{}-'.format(cfg.graph_colors[index])
            index += 1
        plt.plot(range(len(time_series)), time_series, param, label=name)
        plt.xlabel("Year {}".format(time[0].year))
        plt.ylabel(title + " in " + unit)
    ax = plt.gca()
    ax.set_xticks(range(len([i for i in range(len(time)) if time[i].month != time[i - 1].month or i == 0])))
    ax.set_xticklabels(
        [calendar.month_abbr[time[i].month] for i in range(len(time)) if time[i].month != time[i - 1].month or i == 0])
    plt.xticks(rotation=55)
    plt.legend()
    plt.savefig('{}/graphs/{}.pdf'.format(cfg.evaluation_dirs[0], file_name), bbox_inches="tight")
    plt.clf()


def create_evaluation_graphs(gt, outputs):
    data = Dataset('{}/{}{}'.format(cfg.evaluation_dirs[0], cfg.eval_names[0], '_gt.nc'))
    time = data.variables['time']
    time = netCDF4.num2date(time[:], time.units)

    # define dicts for time series
    max_timeseries = {}
    min_timeseries = {}
    mean_timeseries = {}
    rmse_timeseries = {}
    rmse_over_mean_timeseries = {}

    # set GT time series
    max_timeseries['Ground Truth'] = metrics.max_timeseries(gt, time)
    min_timeseries['Ground Truth'] = metrics.min_timeseries(gt, time)
    mean_timeseries['Ground Truth'] = metrics.mean_timeseries(gt, time)

    # define output metrics
    for output_name, output in outputs.items():
        # calculate time series
        max_timeseries[output_name] = metrics.max_timeseries(output, time)
        min_timeseries[output_name] = metrics.min_timeseries(output, time)
        mean_timeseries[output_name] = metrics.mean_timeseries(output, time)
        rmse_timeseries[output_name] = metrics.rmse_timeseries(gt, output, time)
        rmse_over_mean_timeseries[output_name] = metrics.rmse_over_mean_timeseries(gt, output, time)

    # create time series plots
    plot_ts('Maximum', '{}_MaxTS'.format(cfg.eval_names[0]), max_timeseries, time, 'mm/h')
    plot_ts('Minimum', '{}_MinTS'.format(cfg.eval_names[0]), min_timeseries, time, 'mm/h')
    plot_ts('Mean', '{}_MeanTS'.format(cfg.eval_names[0]), mean_timeseries, time, 'mm/h')
    plot_ts('RMSE', '{}_RMSETS'.format(cfg.eval_names[0]), rmse_timeseries, time, 'mm/h')
    plot_ts('AME', '{}_METS'.format(cfg.eval_names[0]), rmse_over_mean_timeseries, time, 'mm/h')


def create_scatter_plots(gt, outputs):
    fld_max = np.max([np.sum(output, axis=(1, 2)) for _, output in outputs.items()])
    timsum_max = np.max([np.sum(output, axis=0) for _, output in outputs.items()])

    for output_name, output in outputs.items():
        gt_sum = np.sum(gt, axis=(1, 2))
        output_sum = np.sum(output, axis=(1, 2))
        plt.scatter(gt_sum, output_sum, s=1)
        plt.plot([np.min(gt_sum), np.max(gt_sum)],
                 [np.min(gt_sum), np.max(gt_sum)],
                 color='red')
        plt.xlabel('Ground truth')
        plt.ylabel(output_name)

        plt.ylim(-fld_max/26, fld_max + fld_max/26)

        plt.savefig('{}/scatter/{}_fldsum_{}.pdf'.format(cfg.evaluation_dirs[0], cfg.eval_names[0], output_name),
                    bbox_inches='tight')
        plt.clf()

        gt_sum = np.sum(gt, axis=0)
        output_sum = np.sum(output, axis=0)
        plt.scatter(gt_sum, output_sum, s=1)
        plt.plot([np.min(gt_sum), np.max(gt_sum)],
                 [np.min(gt_sum), np.max(gt_sum)],
                 color='red')
        plt.xlabel('Ground truth')
        plt.ylabel(output_name)

        plt.ylim(-timsum_max/26, timsum_max + timsum_max/timsum_max)

        plt.savefig('{}/scatter/{}_timsum_{}.pdf'.format(cfg.evaluation_dirs[0], cfg.eval_names[0], output_name),
                    bbox_inches='tight')
        plt.clf()


def create_evaluation_maps(gt, outputs):
    init_font()
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 22})
    timcor_maps = []
    rmse_maps = []
    timcor_names = []
    rmse_names = []

    gt_sum_map = metrics.sum_map(gt)
    sum_maps = []
    sum_names = []
    for output_name, output in outputs.items():
        if cfg.create_sum_maps:
            sum_maps.append(metrics.sum_map(output) - gt_sum_map)
            sum_names.append('Sum {}'.format(output_name))
        if cfg.create_timcor_maps:
            timcor_maps.append(metrics.timcor_map(gt, output))
            timcor_names.append('TimCor {}'.format(output_name))
        if cfg.create_rmse_maps:
            rmse_maps.append(metrics.rmse_map(gt, output))
            rmse_names.append('RMSE {}'.format(output_name))

    map_lists = []
    map_names = []
    if cfg.create_sum_maps:
        map_lists.append(sum_maps)
        map_names.append(sum_names)
    if cfg.create_rmse_maps:
        map_lists.append(rmse_maps)
        map_names.append(rmse_names)
    if cfg.create_timcor_maps:
        map_lists.append(timcor_maps)
        map_names.append(timcor_names)
    for i in range(len(map_lists)):
        minimum = np.min(map_lists[i])
        if 'RMSE' in map_names[i][0]:
            minimum = cfg.min_rmse
            maximum = cfg.max_rmse
        elif 'TimCor' in map_names[i][0]:
            maximum = cfg.max_timcor
        elif 'Sum' in map_names[i][0]:
            minimum = cfg.min_sum
            maximum = cfg.max_sum
        else:
            maximum = np.max(map_lists[i])
        for j in range(len(map_lists[i])):
            # plot and save data
            img = plt.imshow(np.squeeze(map_lists[i][j]), vmin=minimum, vmax=maximum, cmap='RdBu_r', aspect='auto')
            plt.xlabel("km")
            plt.ylabel("km")
            ax = plt.gca()
            ax.set_yticks([i + 12 for i in range(cfg.image_sizes[0]) if i % 100 == 0])
            ax.set_yticklabels([i for i in range(cfg.image_sizes[0]) if i % 100 == 0][::-1])
            ax.set_xticks([i for i in range(cfg.image_sizes[0]) if i % 100 == 0])
            ax.set_xticklabels([i for i in range(cfg.image_sizes[0]) if i % 100 == 0])
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='10%', pad=0.6)
            plt.colorbar(img, cax=cax, orientation='vertical')
            if 'TimCor' not in map_names[i][j]:
                cax.set_xlabel('mm/h', labelpad=20)
                cax.xaxis.set_label_position('bottom')
            plt.savefig('{}/maps/{}_{}.pdf'.format(cfg.evaluation_dirs[0], cfg.eval_names[0], map_names[i][j]), bbox_inches='tight')
            plt.clf()
            plt.close('all')