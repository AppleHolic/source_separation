import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_sound.utils.iterer import single
from torch.nn.init import calculate_gain


class _ComplexConvNd(nn.Module):
    """
    Implement Complex Convolution
    A: real weight
    B: img weight
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.output_padding = output_padding
        self.transposed = transposed

        self.A = self.make_weight(in_channels, out_channels, kernel_size)
        self.B = self.make_weight(in_channels, out_channels, kernel_size)

        self.reset_parameters()

    def make_weight(self, in_ch, out_ch, kernel_size):
        if self.transposed:
            tensor = nn.Parameter(torch.Tensor(in_ch, out_ch // 2, *kernel_size))
        else:
            tensor = nn.Parameter(torch.Tensor(out_ch, in_ch // 2, *kernel_size))
        return tensor

    def reset_parameters(self):
        # init real weight
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.A)

        # init A
        gain = calculate_gain('leaky_relu', 0)
        std = gain / np.sqrt(fan_in)
        bound = np.sqrt(3.0) * std

        with torch.no_grad():
            # TODO: find more stable initial values
            self.A.uniform_(-bound * (1 / (np.pi ** 2)), bound * (1 / (np.pi ** 2)))
            #
            # B is initialized by pi
            # -pi and pi is too big, so it is powed by -1
            self.B.uniform_(-1 / np.pi, 1 / np.pi)


class ComplexConv1d(_ComplexConvNd):
    """
    Complex Convolution 1d
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1):
        kernel_size = single(kernel_size)
        stride = single(stride)
        # edit padding
        padding = padding
        dilation = single(dilation)
        super(ComplexConv1d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                            False, single(0))

    def forward(self, x):
        """
        Implemented complex convolution using combining 'grouped convolution' and 'real / img weight'
        :param x: data (N, C, T) C is concatenated with C/2 real channels and C/2 idea channels
        :return: complex conved result
        """
        # adopt reflect padding
        if self.padding:
            x = F.pad(x, (self.padding, self.padding), 'reflect')

        # forward real
        real_part = F.conv1d(x, self.A, None, stride=self.stride, padding=0,
                             dilation=self.dilation, groups=2)

        # forward idea
        spl = self.in_channels // 2
        weight_B = torch.cat([self.B[:spl].data * (-1), self.B[spl:].data])
        idea_part = F.conv1d(x, weight_B, None, stride=self.stride, padding=0,
                             dilation=self.dilation, groups=2)

        return real_part + idea_part


class ComplexTransposedConv1d(_ComplexConvNd):
    """
    Complex Transposed Convolution 1d
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, dilation=1):
        kernel_size = single(kernel_size)
        stride = single(stride)
        padding = padding
        dilation = single(dilation)
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, True, output_padding)

    def forward(self, x, output_size=None):
        """
        Implemented complex transposed convolution using combining 'grouped convolution' and 'real / img weight'
        :param x: data (N, C, T) C is concatenated with C/2 real channels and C/2 idea channels
        :return: complex transposed convolution result
        """
        # forward real
        if self.padding:
            x = F.pad(x, (self.padding, self.padding), 'reflect')

        real_part = F.conv_transpose1d(x, self.A, None, stride=self.stride, padding=0,
                                       dilation=self.dilation, groups=2)

        # forward idea
        spl = self.out_channels // 2
        weight_B = torch.cat([self.B[:spl] * (-1), self.B[spl:]])
        idea_part = F.conv_transpose1d(x, weight_B, None, stride=self.stride, padding=0,
                                       dilation=self.dilation, groups=2)

        if self.output_padding:
            real_part = F.pad(real_part, (self.output_padding, self.output_padding), 'reflect')
            idea_part = F.pad(idea_part, (self.output_padding, self.output_padding), 'reflect')

        return real_part + idea_part


class ComplexActLayer(nn.Module):
    """
    Activation differently 'real' part and 'img' part
    In implemented DCUnet on this repository, Real part is activated to log space.
    And Phase(img) part, it is distributed in [-pi, pi]...
    """

    def forward(self, x):
        real, img = x.chunk(2, 1)
        return torch.cat([F.leaky_relu_(real), torch.tanh(img) * np.pi], dim=1)
