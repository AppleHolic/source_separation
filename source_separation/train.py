import fire
import os
import torch
import torch.nn as nn
from typing import Tuple
from pytorch_sound.data.meta import voice_bank
from pytorch_sound.models import build_model
from pytorch_sound import settings
from torch.optim.lr_scheduler import MultiStepLR

from source_separation.dataset import get_datasets
from source_separation.trainer import Wave2WaveTrainer, Wave2WaveTrainer2


def main(meta_dir: str, save_dir: str,
         save_prefix: str = '', pretrained_path: str = '',
         model_name: str = 'refine_unet_base', batch_size: int = 128, num_workers: int = 16, fix_len: float = 2.,
         lr: float = 3e-4, betas: Tuple[float] = (0.5, 0.9), weight_decay: float = 0.0,
         max_step: int = 100000, valid_max_step: int = 30, save_interval: int = 1000, log_interval: int = 50,
         grad_clip: float = 0.0, grad_norm: float = 30.0,
         is_audioset: bool = True, milestones: Tuple[int] = None, gamma: float = 0.1):
    # check args
    assert os.path.exists(meta_dir)

    # create model
    model = build_model(model_name).cuda()

    # multi-gpu
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # create optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    if milestones:
        milestones = [int(x) for x in list(milestones)]
        scheduler = MultiStepLR(optimizer, milestones, gamma=gamma)
    else:
        scheduler = None

    # load dataset
    if is_audioset:
        dataset_func = get_datasets
    else:
        dataset_func = voice_bank.get_datasets

    train_loader, valid_loader = dataset_func(
        meta_dir, batch_size=batch_size, num_workers=num_workers,
        fix_len=int(fix_len * settings.SAMPLE_RATE), audio_mask=True
    )

    # train
    Wave2WaveTrainer(
        model, optimizer, train_loader, valid_loader,
        max_step=max_step, valid_max_step=min(valid_max_step, len(valid_loader)), save_interval=save_interval, log_interval=log_interval,
        save_dir=save_dir, save_prefix=save_prefix, grad_clip=grad_clip, grad_norm=grad_norm,
        pretrained_path=pretrained_path, scheduler=scheduler
    ).run()


if __name__ == '__main__':
    fire.Fire(main)
