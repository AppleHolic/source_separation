import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_sound.models import register_model
from pytorch_sound.utils.tensor import concat_complex
from pytorch_sound.models.transforms import STFT

from source_separation.modules import ComplexConv1d, ComplexTransposedConv1d, ComplexActLayer


class ComplexConvBlock(nn.Module):
    """
    Convolution block
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0, layers: int = 4,
                 bn_func=nn.BatchNorm1d, act_func=nn.LeakyReLU, skip_res: bool = False):
        super().__init__()
        # modules
        self.blocks = nn.ModuleList()
        self.skip_res = skip_res

        for idx in range(layers):
            in_ = in_channels if idx == 0 else out_channels
            self.blocks.append(
                nn.Sequential(*[
                    bn_func(in_),
                    act_func(),
                    ComplexConv1d(in_, out_channels, kernel_size, padding=padding),
                ])
            )

    def forward(self, x: torch.tensor) -> torch.tensor:
        temp = x
        for idx, block in enumerate(self.blocks):
            x = block(x)

        if temp.size() != x.size() or self.skip_res:
            return x
        else:
            return x + temp


@register_model('spectrogram_unet')
class SpectrogramUnet(nn.Module):

    def __init__(self, spec_dim: int, hidden_dim: int, filter_len: int, hop_len: int, layers: int = 3,
                 block_layers: int = 3, kernel_size: int = 5, is_mask: bool = False, norm: str = 'bn', act: str = 'tanh'):
        super().__init__()
        self.layers = layers
        self.is_mask = is_mask

        # stft modules
        self.stft = STFT(filter_len, hop_len)

        if norm == 'bn':
            self.bn_func = nn.BatchNorm1d
        elif norm == 'ins':
            self.bn_func = lambda x: nn.InstanceNorm1d(x, affine=True)
        else:
            raise NotImplementedError('{} is not implemented !'.format(norm))

        if act == 'tanh':
            self.act_func = nn.Tanh
            self.act_out = nn.Tanh
        elif act == 'comp':
            self.act_func = ComplexActLayer
            self.act_out = lambda: ComplexActLayer(is_out=True)
        else:
            raise NotImplementedError('{} is not implemented !'.format(act))

        # prev conv
        self.prev_conv = ComplexConv1d(spec_dim * 2, hidden_dim, 1)

        # down
        self.down = nn.ModuleList()
        self.down_pool = nn.MaxPool1d(3, stride=2, padding=1)
        for idx in range(self.layers):
            block = ComplexConvBlock(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2,
                                     bn_func=self.bn_func, act_func=self.act_func, layers=block_layers)
            self.down.append(block)

        # up
        self.up = nn.ModuleList()
        for idx in range(self.layers):
            in_c = hidden_dim if idx == 0 else hidden_dim * 2
            self.up.append(
                nn.Sequential(
                    ComplexConvBlock(in_c, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2,
                                     bn_func=self.bn_func, act_func=self.act_func, layers=block_layers),
                    self.bn_func(hidden_dim),
                    self.act_func(),
                    ComplexTransposedConv1d(hidden_dim, hidden_dim, kernel_size=2, stride=2),
                )
            )

        # out_conv
        self.out_conv = nn.Sequential(
            ComplexConvBlock(hidden_dim * 2, spec_dim * 2, kernel_size=kernel_size,
                             padding=kernel_size // 2, bn_func=self.bn_func, act_func=self.act_func),
            self.bn_func(spec_dim * 2),
            self.act_func()
        )

        # refine conv
        self.refine_conv = nn.Sequential(
            ComplexConvBlock(spec_dim * 4, spec_dim * 2, kernel_size=kernel_size,
                             padding=kernel_size // 2, bn_func=self.bn_func, act_func=self.act_func),
            self.bn_func(spec_dim * 2),
            self.act_func()
        )

    def log_stft(self, wav):
        # stft
        mag, phase = self.stft.transform(wav)
        return torch.log(mag + 1), phase

    def exp_istft(self, log_mag, phase):
        # exp
        mag = np.e ** log_mag - 1
        # istft
        wav = self.stft.inverse(mag, phase)
        return wav

    def adjust_diff(self, x, target):
        size_diff = (target.size()[-1] - x.size()[-1])
        assert size_diff >= 0
        if size_diff > 0:
            x = F.pad(x.unsqueeze(1), (size_diff // 2, size_diff // 2), 'reflect').squeeze(1)
        return x

    def masking(self, mag, phase, origin_mag, origin_phase):
        abs_mag = torch.abs(mag)
        mag_mask = torch.tanh(abs_mag)
        phase_mask = mag / abs_mag

        # masking
        mag = mag_mask * origin_mag
        phase = phase_mask * (origin_phase + phase)
        return mag, phase

    def forward(self, wav):
        # stft
        origin_mag, origin_phase = self.log_stft(wav)
        origin_x = torch.cat([origin_mag, origin_phase], dim=1)

        # prev
        x = self.prev_conv(origin_x)

        # body
        # down
        down_cache = []
        for idx, block in enumerate(self.down):
            x = block(x)
            down_cache.append(x)
            x = self.down_pool(x)

        # up
        for idx, block in enumerate(self.up):
            x = block(x)
            res = F.interpolate(down_cache[self.layers-(idx+1)], size=[x.size()[2]], mode='linear', align_corners=False)
            x = concat_complex(x, res, dim=1)

        # match spec dimension
        x = self.out_conv(x)
        if origin_mag.size(2) != x.size(2):
            x = F.interpolate(x, size=[origin_mag.size(2)], mode='linear', align_corners=False)

        # refine
        x = self.refine_conv(concat_complex(x, origin_x))

        def to_wav(stft):
            mag, phase = stft.chunk(2, 1)
            if self.is_mask:
                mag, phase = self.masking(mag, phase, origin_mag, origin_phase)
            out = self.exp_istft(mag, phase)
            out = self.adjust_diff(out, wav)
            return out

        refine_wav = to_wav(x)

        return refine_wav


@register_model('refine_spectrogram_unet')
class RefineSpectrogramUnet(SpectrogramUnet):

    def __init__(self, spec_dim: int, hidden_dim: int, filter_len: int, hop_len: int, layers: int = 4,
                 block_layers: int = 4, kernel_size: int = 3, is_mask: bool = True, norm: str = 'ins',
                 act: str = 'comp', refine_layers: int = 1, add_spec_results: bool = False):
        super().__init__(spec_dim, hidden_dim, filter_len, hop_len, layers, block_layers,
                         kernel_size, is_mask, norm, act)
        self.add_spec_results = add_spec_results
        # refine conv
        self.refine_conv = nn.ModuleList([
            nn.Sequential(
                ComplexConvBlock(spec_dim * 2, spec_dim * 2, kernel_size=kernel_size,
                                 padding=kernel_size // 2, bn_func=self.bn_func, act_func=self.act_func),
                self.bn_func(spec_dim * 2),
                self.act_func()
            )
        ] * refine_layers)

    def forward(self, wav):
        # stft
        origin_mag, origin_phase = self.log_stft(wav)
        origin_x = torch.cat([origin_mag, origin_phase], dim=1)

        # prev
        x = self.prev_conv(origin_x)

        # body
        # down
        down_cache = []
        for idx, block in enumerate(self.down):
            x = block(x)
            down_cache.append(x)
            x = self.down_pool(x)

        # up
        for idx, block in enumerate(self.up):
            x = block(x)
            res = F.interpolate(down_cache[self.layers - (idx + 1)], size=[x.size()[2]], mode='linear',
                                align_corners=False)
            x = concat_complex(x, res, dim=1)

        # match spec dimension
        x = self.out_conv(x)
        if origin_mag.size(2) != x.size(2):
            x = F.interpolate(x, size=[origin_mag.size(2)], mode='linear', align_corners=False)

        # refine
        for idx, refine_module in enumerate(self.refine_conv):
            x = refine_module(x)
            mag, phase = x.chunk(2, 1)
            mag, phase = self.masking(mag, phase, origin_mag, origin_phase)
            if idx < len(self.refine_conv) - 1:
                x = torch.cat([mag, phase], dim=1)

        # clamp phase
        phase = phase.clamp(-np.pi, np.pi)

        out = self.exp_istft(mag, phase)
        out = self.adjust_diff(out, wav)

        if self.add_spec_results:
            out = (out, mag, phase)

        return out
