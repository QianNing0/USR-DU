import torch
import numpy as np
from torch import nn as nn
import functools
import torch.nn.functional as F
import torch.nn.init as init
from code.utils.registry import ARCH_REGISTRY
from .arch_util import default_init_weights, make_layer, pixel_unshuffle


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        return x + residual


class De_resnet_uncertainty(nn.Module):
    def __init__(self, scale):
        super(De_resnet_uncertainty, self).__init__()
        n_res_blocks = 8

        self.block_input = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.res_blocks = nn.ModuleList([ResidualBlock(64) for _ in range(n_res_blocks)])
        self.scale = scale
        if self.scale == 4:
            self.down_sample = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                nn.PReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                nn.PReLU()
            )
        elif self.scale == 2:
            self.down_sample = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                nn.PReLU(),
            )
        self.block_output = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        # self.b = nn.Parameter(torch.tensor(1.0))
        self.var = nn.Sequential(*[nn.Conv2d(64, 64, kernel_size=3, padding=1),nn.ELU(),nn.Conv2d(64, 64, kernel_size=3, padding=1),nn.ELU(),nn.Conv2d(64, 3, kernel_size=3, padding=1),nn.ELU()])

        self.per_var = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        block = self.block_input(x)
        for res_block in self.res_blocks:
            block = res_block(block)
        if self.down_sample:
            block = self.down_sample(block)
        var = self.var(block)
        per_var = self.per_var(block)
        block = self.block_output(block)
        return [block, var, per_var]


@ARCH_REGISTRY.register()
class RRDBNet_two_vgg19(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, checkpoint_var, checkpoint_U, gc=32, scale=4):
        super(RRDBNet_two_vgg19, self).__init__()
        self.RRDBNet_var = De_resnet_uncertainty(scale)
        self.RRDBNet_U = RRDBNet(in_nc, out_nc, nf, nb, gc)
        
        if checkpoint_var:
            checkpoint_var = torch.load(checkpoint_var, map_location=lambda storage, loc: storage)
            self.RRDBNet_var.load_state_dict(checkpoint_var['gen'], strict=True)
        if checkpoint_U:
            checkpoint_U = torch.load(checkpoint_U, map_location=lambda storage, loc: storage)
            self.RRDBNet_U.load_state_dict(checkpoint_U, strict=True)

        self.up_factor = scale

    def forward(self, x):
        if isinstance(x, list):
            hr = x[1]
            x = x[0]
            with torch.no_grad():
                var = self.RRDBNet_var(hr)[1]
            sam = torch.tensor(np.random.laplace(loc=0.0, scale=np.exp(var.cpu().numpy()), size=var.shape))/255
            x = x + sam.cuda().float()
            x = self.RRDBNet_U(x)
            return x
        else:
            x = self.RRDBNet_U(x)

            return x

