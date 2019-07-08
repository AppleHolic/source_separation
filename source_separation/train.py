import fire
import os
import torch
import torch.nn as nn
from typing import Tuple
from pytorch_sound.data.meta.voice_bank import get_datasets
from pytorch_sound.models import build_model
from pytorch_sound import settings
from source_separation.trainer import Wave2WaveTrainer


def main(meta_dir: str, save_dir: str,
         save_prefix: str = '', pretrained_path: str = '',
         model_name: str = '', batch_size: int = 16, num_workers: int = 16, fix_len: float = 2.0,
         lr: float = 1e-3, betas: Tuple[float] = (0.9, 0.999), weight_decay: float = 0.0,
         max_step: int = 200000, valid_max_step: int = 30, save_interval: int = 1000, log_interval: int = 100,
         grad_clip: float = 0.0, grad_norm: float = 10.0):
    # check args
    assert os.path.exists(meta_dir)

    # create model
    model = build_model(model_name).cuda()

    # multi-gpu
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

    # load dataset
    train_loader, valid_loader = get_datasets(
        meta_dir, batch_size=batch_size, num_workers=num_workers,
        fix_len=int(fix_len * settings.SAMPLE_RATE), audio_mask=True
    )

    # train
    Wave2WaveTrainer(
        model, optimizer, train_loader, valid_loader,
        max_step=max_step, valid_max_step=valid_max_step, save_interval=save_interval, log_interval=log_interval,
        save_dir=save_dir, save_prefix=save_prefix, grad_clip=grad_clip, grad_norm=grad_norm,
        pretrained_path=pretrained_path
    ).run()


if __name__ == '__main__':
    fire.Fire(main)
