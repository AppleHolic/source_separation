import torch
import torch.nn as nn
from typing import Tuple, Dict
from pytorch_sound.trainer import Trainer, LogType


class Wave2WaveTrainer(Trainer):

    def __init__(self, model: nn.Module, optimizer,
                 train_dataset, valid_dataset,
                 max_step: int, valid_max_step: int, save_interval: int, log_interval: int,
                 save_dir: str, save_prefix: str = '',
                 grad_clip: float = 0.0, grad_norm: float = 0.0,
                 pretrained_path: str = None):
        super().__init__(model, optimizer, train_dataset, valid_dataset,
                         max_step, valid_max_step, save_interval, log_interval, save_dir, save_prefix,
                         grad_clip, grad_norm, pretrained_path)
        # loss
        self.mse_loss = nn.MSELoss()

    def forward(self, noise, clean, speaker, txt, mask, is_logging: bool = False) -> Tuple[torch.Tensor, Dict]:
        # forward
        res = self.model(noise)
        if isinstance(res, tuple):
            clean_hat, *_ = res
        else:
            clean_hat = res

        # calc loss
        loss = self.mse_loss(clean_hat, clean[..., :clean_hat.size(-1)])

        meta = {
            'loss': (loss.item(), LogType.SCALAR),
            'clean_hat.audio': (clean_hat[0], LogType.AUDIO),
            'clean.audio': (clean[0], LogType.AUDIO),
            'noise.audio': (noise[0], LogType.AUDIO),
            'clean_hat.plot': (clean_hat[0], LogType.PLOT),
            'clean.plot': (clean[0], LogType.PLOT),
            'noise.plot': (noise[0], LogType.PLOT),
        }

        return loss, meta
