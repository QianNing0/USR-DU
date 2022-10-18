import torch
import torch.nn as nn


def make_model(args, parent=False):
    return De_resnet_uncertainty(args)

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
    def __init__(self, args):
        super(De_resnet_uncertainty, self).__init__()
        n_res_blocks = args.n_res_blocks
        scale = int(args.scale)

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
