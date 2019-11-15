import fire
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any
from pytorch_sound.data.meta import voice_bank, dsd100
from pytorch_sound.data.meta.dsd100 import DSD100Meta
from pytorch_sound.data.meta.voice_bank import VoiceBankMeta
from pytorch_sound.models import build_model
from torch.optim.lr_scheduler import MultiStepLR

from source_separation.dataset import get_datasets
from source_separation.trainer import Wave2WaveTrainer


def main(args: Dict[str, Any]):
    save_prefix = '_'.join(['{}-{}'.format(key, val) for key, val in args.items()])
    return _main(save_prefix=save_prefix, **args)


def _main(meta_dir: str,
          save_prefix: str = '',
          model_name: str = 'refine_unet_base',   # or refine_spectrogram_unet
          save_dir: str = 'savedir', batch_size: int = 128, num_workers: int = 16, fix_len: float = 2.,
          lr: float = 5e-4, beta1: float = 0.5, beta2: float = 0.9, weight_decay: float = 0.0,
          max_step: int = 100000, valid_max_step: int = 30, save_interval: int = 1000, log_interval: int = 100,
          grad_clip: float = 0.0, grad_norm: float = 30.0, milestones: Tuple[int] = None, gamma: float = 0.2,
          is_augment: bool = True, is_dsd: bool = False,
          # model args
          hidden_dim: int = 768, filter_len: int = 512, hop_len: int = 64,
          block_layers: int = 4, layers: int = 4, kernel_size: int = 3, norm: str = 'ins', act: str = 'comp',
          refine_layers: int = 1,
          ):
    betas = beta1, beta2

    # setup model args
    model_args = {
        'hidden_dim': hidden_dim,
        'filter_len': filter_len,
        'hop_len': hop_len,
        'spec_dim': filter_len // 2 + 1,
        'block_layers': block_layers,
        'layers': layers,
        'kernel_size': kernel_size,
        'norm': norm,
        'refine_layers': refine_layers,
        'act': act
    }

    # create model
    model = build_model(model_name, extra_kwargs=model_args).cuda()

    # multi-gpu
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

    if milestones:
        milestones = [int(x) for x in list(milestones)]
        scheduler = MultiStepLR(optimizer, milestones, gamma=gamma)
    else:
        scheduler = None

    # adopt dsd100 case
    if is_dsd:
        sr = 44100
        if is_augment:
            dataset_func = get_datasets
            meta_cls = DSD100Meta
        else:
            dataset_func = dsd100.get_datasets
    else:
        sr = 22050
        # load dataset
        if is_augment:
            dataset_func = get_datasets
            meta_cls = VoiceBankMeta
        else:
            dataset_func = voice_bank.get_datasets

    train_loader, valid_loader = dataset_func(
        meta_dir, batch_size=batch_size, num_workers=num_workers, meta_cls=meta_cls,
        fix_len=int(fix_len * sr), audio_mask=True
    )

    # train
    loss = Wave2WaveTrainer(
        model, optimizer, train_loader, valid_loader,
        max_step=max_step, valid_max_step=min(valid_max_step, len(valid_loader)), save_interval=save_interval,
        log_interval=log_interval,
        save_dir=save_dir, save_prefix=save_prefix, grad_clip=grad_clip, grad_norm=grad_norm,
        pretrained_path='', scheduler=scheduler, sr=sr
    ).run()

    return {
        'loss': loss,
        'status': 'ok',
    }


if __name__ == '__main__':
    fire.Fire(_main)
