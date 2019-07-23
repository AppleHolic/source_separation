import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_sound.models import register_model
from pytorch_sound.utils.calculate import conv_same_padding


class ConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float = 0.1, layers: int = 5):
        super().__init__()
        # modules
        self.blocks = nn.ModuleList()
        self.pad_size = 0

        for idx in range(layers):

            in_c = in_channels if idx == 0 else out_channels

            dilation = 2 ** idx
            padding = conv_same_padding(kernel_size, stride=1, dilation=dilation)

            self.blocks.append(
                nn.Sequential(*[
                    nn.Dropout(dropout),
                    nn.Tanh(),
                    nn.Conv1d(in_c, out_channels, kernel_size, padding=padding, dilation=dilation)
                ])
            )

            self.pad_size += padding

    def forward(self, x):
        # save temp
        temp = x
        for idx, block in enumerate(self.blocks):
            x = block(x)

        if temp.size() != x.size():
            return x
        else:
            return temp + x


@register_model('wave_unet')
class WaveUNet(nn.Module):

    def __init__(self, hidden_dim: int, kernel_size: int, dropout: float, layers: int, block_layers: int):
        super().__init__()
        self.layers = layers

        # prev conv
        self.prev_conv = nn.Conv1d(1, hidden_dim, 1)

        # down
        self.down = nn.ModuleList([
            ConvBlock(hidden_dim, hidden_dim, kernel_size=kernel_size, layers=block_layers)
        ] * self.layers)
        self.down_pool = nn.MaxPool1d(kernel_size, stride=2, padding=kernel_size // 2)

        # up
        self.up = nn.ModuleList()
        for idx in range(self.layers):
            in_c = hidden_dim if idx == 0 else hidden_dim * 2
            self.up.append(
                nn.Sequential(*[
                    ConvBlock(in_c, hidden_dim, kernel_size=kernel_size, layers=block_layers),
                    nn.Dropout(dropout),
                    nn.Tanh(),
                    nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=2, stride=2)
                ])
            )

        # out_conv
        self.out_conv = nn.Sequential(
            ConvBlock(hidden_dim * 2, hidden_dim, kernel_size=kernel_size, layers=block_layers),
            nn.Dropout(dropout),
            nn.Tanh(),
            nn.Conv1d(hidden_dim, 1, 1),
            nn.Tanh()
        )

    def forward(self, wav):
        temp = wav.unsqueeze(1)
        x = self.prev_conv(temp)

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
            x = torch.cat([x, F.interpolate(down_cache[self.layers-(idx+1)], size=[x.size()[2]], mode='linear')], dim=1)

        x = self.out_conv(x)

        # match size
        size_diff = (wav.size()[-1] - x.size()[-1])
        if size_diff > 0:
            x = F.pad(x, (size_diff // 2, size_diff // 2), 'reflect')
        elif size_diff < 0:
            x = x[..., :wav.size()[-1]]

        return x.squeeze(1)
