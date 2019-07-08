import torch
from typing import Tuple, Dict
from pytorch_sound.trainer import Trainer, LogType


class Wave2WaveTrainer(Trainer):

    def mse_loss(self, clean_hat, clean, mask):
        mask = mask[..., :clean_hat.size()[-1]]
        wav_len = torch.sum(mask, dim=1, keepdim=True)
        clean = clean[..., :clean_hat.size()[-1]]
        loss = (clean_hat * mask - clean * mask) ** 2
        # reduce
        loss = torch.mean(torch.sum(loss, dim=1, keepdim=True) / wav_len)
        return loss

    def forward(self, noise, clean, speaker, txt, mask, is_logging: bool = False) -> Tuple[torch.Tensor, Dict]:
        # forward
        clean_hat = self.model(noise)

        # calc loss
        loss = self.mse_loss(clean_hat, clean, mask)

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
