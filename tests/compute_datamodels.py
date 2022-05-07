import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch as ch

from fastargs import Param, Section, get_current_config
from fastargs.decorators import param, section

from ffcv.fields.decoders import IntDecoder, NDArrayDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import Squeeze, ToDevice, ToTensor

from fast_l1 import regressor

Section('data', 'source data info').params(
    data_path=Param(str, 'Path to beton file', required=True),
    num_train=Param(int, 'Number of models for training', required=True),
    num_val=Param(int, 'Number of models for validation', required=True),
    seed=Param(int, 'Random seed for picking validation set')
    # split=Param(And(str, OneOf(['train', 'test'])),
    # 'Which data we are computing on', required=True),
)

Section('cfg', 'arguments to give the writer').params(
    k=Param(int, 'Number of lambdas on the regularization path',
            required=True),
    lr=Param(float, 'Learning rate to use', default=0.01),
    eps=Param(float, '(min lambda) / (max lambda)', default=1e-5),
    batch_size=Param(int, 'Batch size for regression', required=True),
    out_dir=Param(str, 'Where to write', required=True),
    num_workers=Param(int, 'Num of workers to use for dataloading', default=16)
)

Section('early_stopping', 'arguments specific to early stopping').params(
    check_every=Param(int, 'How often to check for improvement', default=2),
    eps=Param(float, 'Improvement required at every check', default=1e-5)
)


@param('data.data_path')
@param('cfg.num_workers')
@param('cfg.batch_size')
def make_loader(subset, data_path=None, num_workers=None,
                drop_last=True, batch_size: int = 0) -> Loader:
    assert len(subset) % batch_size == 0, \
        f'Batch size ({batch_size}) should divide dataset size ({len(subset)})'
    return Loader(data_path,
                  batch_size=batch_size,
                  num_workers=num_workers,
                  order=OrderOption.RANDOM,
                  indices=subset,
                  drop_last=drop_last,
                  os_cache=True,
                  pipelines={
                      'mask': [NDArrayDecoder(),
                               ToTensor(),
                               ToDevice(ch.device('cuda:0'))],
                      'targets': [NDArrayDecoder(),
                                  ToTensor(),
                                  ToDevice(ch.device('cuda:0'))],
                      'idx': [IntDecoder(),
                              ToTensor(),
                              Squeeze(),
                              ToDevice(ch.device('cuda:0'))]
                  }, recompile=False)


@param('data.num_train')
@param('data.num_val')
def make_loaders(num_train: int = -1, num_val: int = -1):
    return make_loader(subset=np.arange(num_train)), \
           make_loader(subset=np.arange(num_train, num_train + num_val)), \
           make_loader(subset=np.arange(num_train + num_val))


@section('cfg')
@param('lr')
@param('k')
@param('eps')
@param('out_dir')
@section('early_stopping')
@param('check_every', alias='early_stop_freq')
@param('eps', alias='early_stop_eps')
def main(lr: float, k: int, eps: float,
         out_dir: str,
         early_stop_freq: int,
         early_stop_eps: float):
    train_loader, val_loader, full_loader = make_loaders()
    max_lam = regressor.calc_max_lambda(train_loader)

    n_features = train_loader.reader.handlers['mask'].shape[0]
    n_targets = train_loader.reader.handlers['targets'].shape[0]

    weight = ch.zeros(n_features, n_targets).cuda()
    bias = ch.zeros(n_targets).cuda()

    assert not os.path.exists(out_dir)
    log_path = Path(out_dir) / 'regularization_path/'
    os.makedirs(log_path)
    best_lam = \
        regressor.train_saga(weight,
                             bias,
                             train_loader,
                             val_loader,
                             lr=lr,
                             start_lams=max_lam,
                             lam_decay=np.exp(np.log(eps)/k),
                             num_lambdas=k,
                             early_stop_freq=early_stop_freq,
                             early_stop_eps=early_stop_eps,
                             logdir=str(log_path))

    ch.cuda.empty_cache()
    regressor.train_saga(weight,
                         bias,
                         full_loader,
                         None,
                         lr=lr,
                         start_lams=best_lam,
                         lam_decay=1.,
                         num_lambdas=1,
                         early_stop_freq=early_stop_freq,
                         early_stop_eps=early_stop_eps,
                         logdir='/tmp')
    ch.save({
        'weight': weight.cpu(),
        'bias':  bias.cpu(),
        'lam': best_lam.cpu()
    }, Path(out_dir) / 'datamodels.pt')


if __name__ == '__main__':
    config = get_current_config()
    parser = ArgumentParser(description='Datamodel regression')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()
    main()
