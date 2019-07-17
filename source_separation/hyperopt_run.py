import fire
import json
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any
from pytorch_sound.data.meta import voice_bank
from pytorch_sound.models import build_model
from pytorch_sound import settings
from source_separation.dataset import get_datasets
from source_separation.trainer import Wave2WaveTrainer


def main(args: Dict[str, Any]):
    return _main(**args)


def _main(model_name: str = 'spectrogram_unet_comp',   # or refine_spectrogram_unet
         save_dir: str = 'savedir', batch_size: int = 128, num_workers: int = 16, fix_len: float = 2.,
         lr: float = 1e-4, betas: Tuple[float] = (0.5, 0.9), weight_decay: float = 0.0,
         max_step: int = 70000, valid_max_step: int = 30, save_interval: int = 1000, log_interval: int = 100,
         grad_clip: float = 0.0, grad_norm: float = 30.0,
         is_audioset: bool = True,
         # model args
         hidden_dim: int = 384, filter_len: int = 1024, hop_len: int = 256,
         block_layers: int = 4, layers: int = 5, kernel_size: int = 3, norm: str = 'bn', refine_layers: int = 1,
         ):
    # check args
    config_path = '../assets/config.json'
    with open(config_path, 'r') as r:
        configs = json.load(r)
        meta_dir = configs['meta_dir']

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
        'refine_layers': refine_layers
    }

    # create model
    model = build_model(model_name, extra_kwargs=model_args).cuda()

    # multi-gpu
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

    # load dataset
    if is_audioset:
        dataset_func = get_datasets
    else:
        dataset_func = voice_bank.get_datasets

    train_loader, valid_loader = dataset_func(
        meta_dir, batch_size=batch_size, num_workers=num_workers,
        fix_len=int(fix_len * settings.SAMPLE_RATE), audio_mask=True
    )

    # model prefix
    save_prefix = '_'.join(['{}-{}'.format(key, val) for key, val in model_args.items()])

    # train
    loss = Wave2WaveTrainer(
        model, optimizer, train_loader, valid_loader,
        max_step=max_step, valid_max_step=valid_max_step, save_interval=save_interval, log_interval=log_interval,
        save_dir=save_dir, save_prefix=save_prefix, grad_clip=grad_clip, grad_norm=grad_norm,
        pretrained_path=''
    ).run()

    return {
        'loss': loss,
        'ok': True
    }


if __name__ == '__main__':
    fire.Fire(_main)
