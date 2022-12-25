import os
import random

import h5py
import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset, Sampler

from .netcdfchecker import dataset_formatter
from .normalizer import img_normalization
from .. import config as cfg


def load_steadymask(path, mask_name, data_type, device):
    if mask_name is None:
        return None
    else:
        steady_mask, _ = load_netcdf(path, [mask_name], [data_type])
        return torch.from_numpy(steady_mask[0]).to(device)


class InfiniteSampler(Sampler):
    def __init__(self, num_samples, data_source=None):
        super().__init__(data_source)
        self.num_samples = num_samples

    def __iter__(self):
        return iter(self.loop())

    def __len__(self):
        return 2 ** 31

    def loop(self):
        i = 0
        order = np.random.permutation(self.num_samples)
        while True:
            yield order[i]
            i += 1
            if i >= self.num_samples:
                order = np.random.permutation(self.num_samples)
                i = 0


def nc_loadchecker(filename, data_type, image_size, keep_dss=False):
    basename = filename.split("/")[-1]

    if not os.path.isfile(filename):
        print('File {} not found.'.format(filename))

    try:
        # We use load_dataset instead of open_dataset because of lazy transpose
        ds = xr.load_dataset(filename, decode_times=False)
    except Exception:
        raise ValueError('Impossible to read {}.'
                         '\nPlease, check that it is a netCDF file and it is not corrupted.'.format(basename))

    ds1 = dataset_formatter(ds, data_type, image_size, basename)

    if keep_dss:
        dtype = ds[data_type].dtype
        ds = ds.drop_vars(data_type)
        ds[data_type] = np.empty(0, dtype=dtype)
        return [ds, ds1], [ds1[data_type].values]
    else:
        return None, [ds1[data_type].values]


def load_netcdf(path, data_names, data_types, keep_dss=False):
    if data_names is None:
        return None, None
    else:
        ndata = len(data_names)
        assert ndata == len(data_types)

        dss, data = nc_loadchecker('{}{}'.format(path, data_names[0]), data_types[0], cfg.image_sizes[0],
                                   keep_dss=keep_dss)
        lengths = [len(data[0])]
        for i in range(1, ndata):
            data += nc_loadchecker('{}{}'.format(path, data_names[i]), data_types[i], cfg.image_sizes[0])[1]
            lengths.append(len(data[-1]))

        if cfg.img_index is None:
            assert len(set(lengths)) == 1

        if keep_dss:
            return dss, data, lengths[0]
        else:
            return data, lengths[0]


class NetCDFLoader(Dataset):
    def __init__(self, data_root, img_names, mask_root, mask_names, split, data_types, time_steps):
        super(NetCDFLoader, self).__init__()
        self.split = split
        self.data_types = data_types
        self.img_names = img_names
        self.mask_names = mask_names
        self.lstm_steps = time_steps
        self.prev_next_steps = 0

        if split == 'train':
            self.data_path = '{:s}/train/'.format(data_root)
        elif split == 'test' or split == 'infill':
            self.data_path = '{:s}/test/'.format(data_root)
        elif split == 'val':
            self.data_path = '{:s}/val/'.format(data_root)
        self.mask_path = mask_root

        # define image and mask lenghts
        self.img_lengths = {}
        self.mask_lengths = {}
        self.bounds = None
        assert len(img_names) == len(mask_names) == len(data_types)
        for i in range(len(img_names)):
            self.init_dataset(img_names[i], mask_names[i], data_types[i])

    def init_dataset(self, img_name, mask_name, data_type):
        # set img and mask length
        img_file = h5py.File('{}{}'.format(self.data_path, img_name), 'r')
        img_data = img_file.get(data_type)
        mask_file = h5py.File('{}{}'.format(self.mask_path, mask_name), 'r')
        mask_data = mask_file.get(data_type)
        self.img_lengths[img_name] = img_data.shape[0]
        self.mask_lengths[mask_name] = mask_data.shape[0]

        # if infill, check if img length matches mask length
        if self.split == 'infill':
            assert img_data.shape[0] == mask_data.shape[0]

    def load_data(self, file, data_type, indices):
        # open netcdf file
        h5_data = file.get(data_type)
        try:
            if h5_data.ndim == 4:
                total_data = torch.from_numpy(h5_data[indices, 0, :, :])
            else:
                total_data = torch.from_numpy(h5_data[indices, :, :])
        except TypeError:
            # get indices that occur more than once
            unique, counts = np.unique(indices, return_counts=True)
            copy_indices = [(index, counts[index] - 1) for index in range(len(counts)) if counts[index] > 1]
            if h5_data.ndim == 4:
                total_data = torch.from_numpy(h5_data[unique, 0, :, :])
            else:
                total_data = torch.from_numpy(h5_data[unique, :, :])
            if unique[copy_indices[0][0]] == 0:
                total_data = torch.cat([torch.stack(copy_indices[0][1] * [total_data[copy_indices[0][0]]]), total_data])
            else:
                total_data = torch.cat([total_data, torch.stack(copy_indices[0][1] * [total_data[copy_indices[0][0]]])])
        return total_data

    def get_single_item(self, index, img_name, mask_name, data_type):
        if self.lstm_steps == 0:
            prev_steps = next_steps = self.prev_next_steps
        else:
            prev_steps = next_steps = self.lstm_steps

        # define range of lstm or prev-next steps -> adjust, if out of boundaries
        img_indices = np.array(list(range(index - prev_steps, index + next_steps + 1)))
        img_indices[img_indices < 0] = 0
        img_indices[img_indices > self.img_lengths[img_name] - 1] = self.img_lengths[img_name] - 1
        if self.split == 'infill':
            mask_indices = img_indices
        else:
            mask_indices = []
            mask_index = random.randint(0, self.mask_lengths[mask_name] - 1)
            for j in range(prev_steps + next_steps + 1):
                if not cfg.select_continuous_masks:
                    mask_index = random.randint(0, self.mask_lengths[mask_name] - 1)
                mask_indices.append(mask_index)
            mask_indices = sorted(mask_indices)

        # load data from ranges
        img_file = h5py.File('{}{}'.format(self.data_path, img_name), 'r')
        mask_file = h5py.File('{}{}'.format(self.mask_path, mask_name), 'r')
        images = self.load_data(img_file, data_type, img_indices)
        masks = self.load_data(mask_file, data_type, mask_indices)

        # stack to correct dimensions
        if self.lstm_steps == 0:
            images = torch.cat([images], dim=0).unsqueeze(0)
            masks = torch.cat([masks], dim=0).unsqueeze(0)
        else:
            images = images.unsqueeze(1)
            masks = masks.unsqueeze(1)
        return images, masks

    def __getitem__(self, index):
        image, mask = self.get_single_item(index, self.img_names[0], self.mask_names[0], self.data_types[0])
        images = []
        masks = []
        for i in range(1, len(self.data_types)):
            img, m = self.get_single_item(index, self.img_names[i], self.mask_names[i], self.data_types[i])
            images.append(img)
            masks.append(m)
        if images and masks:
            images = torch.cat(images, dim=1)
            masks = torch.cat(masks, dim=1)
            return mask * image, mask, image, masks * images, masks, images
        else:
            return mask * image, mask, image, torch.tensor([]), torch.tensor([]), torch.tensor([])

    def __len__(self):
        return self.img_lengths[self.img_names[0]]
