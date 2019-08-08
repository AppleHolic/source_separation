import torch
import torch.nn as nn
from typing import Tuple, Dict

from pytorch_sound.models.sound import InversePreEmphasis
from pytorch_sound.trainer import Trainer, LogType


class Wave2WaveTrainer(Trainer):

    def __init__(self, model: nn.Module, optimizer,
                 train_dataset, valid_dataset,
                 max_step: int, valid_max_step: int, save_interval: int, log_interval: int,
                 save_dir: str, save_prefix: str = '',
                 grad_clip: float = 0.0, grad_norm: float = 0.0,
                 pretrained_path: str = None, scheduler: torch.optim.lr_scheduler._LRScheduler = None):
        super().__init__(model, optimizer, train_dataset, valid_dataset,
                         max_step, valid_max_step, save_interval, log_interval, save_dir, save_prefix,
                         grad_clip, grad_norm, pretrained_path, scheduler=scheduler)
        # loss
        self.mse_loss = nn.MSELoss()
        # pytorch inverse preemphasis module
        self.inv_preemp = InversePreEmphasis().cuda()

    def l1_loss(self, clean_hat, clean):
        return torch.abs(clean_hat - clean).mean()

    def forward(self, noise, clean, speaker, txt, mask, is_logging: bool = False) -> Tuple[torch.Tensor, Dict]:
        # forward
        res = self.model(noise)
        if isinstance(res, tuple):
            clean_hat, *_ = res
        else:
            clean_hat = res

        # calc loss
        # loss = self.mse_loss(clean_hat, clean[..., :clean_hat.size(-1)])
        loss = self.l1_loss(clean_hat, clean[..., :clean_hat.size(-1)])

        if is_logging:
            inf = torch.cat([clean_hat[:1], clean[:1], noise[:1]], dim=0).unsqueeze(1)
            inf = self.inv_preemp(inf).squeeze(1)
            clean_hat, clean, noise = map(lambda x: x.squeeze(), inf.chunk(3, 0))

            meta = {
                'loss': (loss.item(), LogType.SCALAR),
                'clean_hat.audio': (clean_hat, LogType.AUDIO),
                'clean.audio': (clean, LogType.AUDIO),
                'noise.audio': (noise, LogType.AUDIO),
                'clean_hat.plot': (clean_hat, LogType.PLOT),
                'clean.plot': (clean, LogType.PLOT),
                'noise.plot': (noise, LogType.PLOT),
            }
        else:
            meta = {}

        return loss, meta
