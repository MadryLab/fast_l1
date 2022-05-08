import numpy as np
import os
from typing import Optional, Sequence
from torch.utils.data import Dataset
from ffcv.writer import DatasetWriter
from ffcv.fields import NDArrayField, IntField

from argparse import ArgumentParser
from fastargs import Section, Param
from fastargs.decorators import param
from fastargs import get_current_config

Section('cfg', 'arguments to give the writer').params(
    data_dir=Param(str, 'Where to find the mmap datasets', required=True),
    out_path=Param(str, 'Where to write the FFCV dataset', required=True),
    x_name=Param(str, 'What portion of the data to write', default='masks'),
    y_name=Param(str, 'What portion of the data to write', required=True)
)


class RegressionDataset(Dataset):
    def __init__(self, *, masks_path: str, y_path: str,
                 subset: Optional[Sequence[int]] = None):
        super().__init__()
        self.masks_fp = np.lib.format.open_memmap(masks_path, mode='r')
        self.x_dtype = self.masks_fp.dtype
        self.y_vals_fp = np.lib.format.open_memmap(y_path, mode='r')
        self.y_dtype = self.y_vals_fp.dtype
        self.subset = subset or range(self.masks_fp.shape[0])

    def __getitem__(self, idx):
        inds = self.subset[idx]
        x_val = self.masks_fp[inds]
        y_val = self.y_vals_fp[inds].astype('float32')
        return x_val, y_val, inds

    def shape(self):
        return self.masks_fp.shape[1], self.y_vals_fp.shape[1]

    def __len__(self):
        return len(self.subset)


@param('cfg.data_dir')
@param('cfg.out_path')
@param('cfg.x_name')
@param('cfg.y_name')
def write_dataset(data_dir: str, out_path: str, x_name: str, y_name: str):
    ds = RegressionDataset(
            masks_path=os.path.join(data_dir, f'{x_name}.npy'),
            y_path=os.path.join(data_dir, f'{y_name}.npy'))

    x_dim, y_dim = ds.shape()
    writer = DatasetWriter(out_path, {
        'mask': NDArrayField(dtype=ds.x_dtype, shape=(x_dim,)),
        'targets': NDArrayField(dtype=ds.y_dtype, shape=(y_dim,)),
        'idx': IntField()
    })

    writer.from_indexed_dataset(ds)


if __name__ == '__main__':
    config = get_current_config()
    parser = ArgumentParser(description='Fast imagenet training')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()
    write_dataset()
