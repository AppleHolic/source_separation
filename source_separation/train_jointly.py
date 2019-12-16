import fire
import torch
import torch.nn as nn
from typing import Tuple
from pytorch_sound.data.meta.dsd100 import DSD100Meta
from pytorch_sound.data.meta.musdb18 import MUSDB18Meta
from pytorch_sound.data.meta.voice_bank import VoiceBankMeta
from pytorch_sound.models import build_model
from torch.optim.lr_scheduler import MultiStepLR

from source_separation.dataset import get_concated_datasets
from source_separation.trainer import Wave2WaveTrainer, LossMixingTrainer


def main(vb_meta_dir: str, music_meta_dir: str, save_dir: str, save_prefix: str, pretrained_path: str = '',
         model_name: str = 'refine_unet_larger', batch_size: int = 128, num_workers: int = 16, fix_len: float = 2.,
         lr: float = 5e-4, betas: Tuple[float] = (0.5, 0.9), weight_decay: float = 0.0,
         max_step: int = 200000, valid_max_step: int = 50, save_interval: int = 1000, log_interval: int = 50,
         grad_clip: float = 0.0, grad_norm: float = 30.0,
         milestones: Tuple[int] = None, gamma: float = 0.1, sample_rate: int = 44100, music_data_name: str = 'dsd100',
         mix_loss: bool = False):

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

    # choose music source dataset
    if music_data_name == 'musdb18':
        MusicMeta = MUSDB18Meta
    elif music_data_name == 'dsd100':
        MusicMeta = DSD100Meta
    else:
        raise NotImplementedError(f'{music_data_name} is not implemented ! choose one in [\'musdb18\', \'dsd100\']')

    # make metas
    meta_dir_list = [vb_meta_dir, music_meta_dir]
    meta_cls_list = [VoiceBankMeta, MusicMeta]
    meta_dir_list, meta_cls_list = map(list, zip(*[(d, cls) for d, cls in zip(meta_dir_list, meta_cls_list) if d]))

    train_loader, valid_loader = get_concated_datasets(
        meta_dir_list, batch_size=batch_size, num_workers=num_workers, meta_cls_list=meta_cls_list,
        fix_len=int(fix_len * sample_rate), audio_mask=True
    )

    if mix_loss:
        trainer = LossMixingTrainer
    else:
        trainer = Wave2WaveTrainer

    # train
    trainer(
        model, optimizer, train_loader, valid_loader,
        max_step=max_step, valid_max_step=min(valid_max_step, len(valid_loader)), save_interval=save_interval,
        log_interval=log_interval,
        save_dir=save_dir, save_prefix=save_prefix, grad_clip=grad_clip, grad_norm=grad_norm,
        pretrained_path=pretrained_path, scheduler=scheduler, sr=sample_rate
    ).run()


if __name__ == '__main__':
    fire.Fire(main)
