import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import weight_norm, remove_weight_norm

from configs import TrainConfig


class ResBlock(nn.Module):
    def __init__(self, train_config: TrainConfig, in_channels, kernel_size=3):
        super(ResBlock, self).__init__()

        self.convs = nn.ModuleList([])
        self.relu_ratio = train_config.relu

        for stride, dilation in train_config.D_r:
            new_convs = nn.ModuleList([
                weight_norm(nn.Conv1d(in_channels, in_channels, kernel_size,
                                      stride=stride, dilation=dilation,
                                      padding=int((kernel_size * dilation - dilation) / 2))),
                weight_norm(nn.Conv1d(in_channels, in_channels, kernel_size,
                                      stride=stride, dilation=dilation,
                                      padding=int((kernel_size * dilation - dilation) / 2)))
            ])
            new_convs[0].weight.data.normal_(0, 0.01)
            new_convs[1].weight.data.normal_(0, 0.01)
            self.convs.append(new_convs)

    def forward(self, x):
        for conv_pair in self.convs:
            out = x
            for conv in conv_pair:
                out = F.leaky_relu(out, self.relu_ratio)
                out = conv(out)
            x = x + out
        return x

    def remove_weight_norm(self):
        for i in range(len(self.convs)):
            remove_weight_norm(self.convs[i][0])
            remove_weight_norm(self.convs[i][1])


class MRF(nn.Module):
    def __init__(self, train_config, in_channels):
        super(MRF, self).__init__()
        self.resblocks = nn.ModuleList([])

        for i in range(len(train_config.resblock_kernel_sizes)):
            self.resblocks.append(ResBlock(train_config, in_channels,
                                           train_config.resblock_kernel_sizes[i]))

    def forward(self, x):
        out = None
        for resblock in self.resblocks:
            if out is None:
                out = resblock(x)
            else:
                out += resblock(x)
        out = out / len(self.resblocks)
        return out

    def remove_weight_norm(self):
        for i in range(len(self.resblocks)):
            self.resblocks[i].remove_weight_norm()


class Generator(nn.Module):
    def __init__(self, train_config: TrainConfig):
        super(Generator, self).__init__()
        self.relu_coef = train_config.relu

        self.pre_conv = weight_norm(nn.Conv1d(80, train_config.upsample_initial_channel,
                                              7, 1, padding=3))
        self.pre_conv.weight.data.normal_(0, 0.01)

        self.blocks = nn.ModuleList([])
        for i in range(len(train_config.resblock_kernel_sizes)):
            new_block = nn.ModuleList([])
            new_block.append(weight_norm(
                nn.ConvTranspose1d(train_config.upsample_initial_channel // (2 ** i),
                                   train_config.upsample_initial_channel // (2 ** (i + 1)),
                                   train_config.upsample_kernel_sizes[i],
                                   stride=train_config.upsample_rates[i],
                                   padding=(train_config.upsample_kernel_sizes[i] - train_config.upsample_rates[i]) // 2)
            ))
            new_block[0].weight.data.normal_(0, 0.01)
            new_block.append(
                MRF(train_config, train_config.upsample_initial_channel // (2 ** (i + 1)))
            )
            self.blocks.append(new_block)

        self.post_conv = weight_norm(nn.Conv1d(
            train_config.upsample_initial_channel // (2 ** len(train_config.resblock_kernel_sizes)),
            1, 7, padding=3
        ))
        self.post_conv.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.pre_conv(x)

        for block in self.blocks:
            out = F.leaky_relu(out, self.relu_coef)
            out = block[0](out)
            out = block[1](out)

        out = F.leaky_relu(out, self.relu_coef)
        out = self.post_conv(out)
        out = torch.tahn(out)

        return out

    def remove_weight_norm(self):
        for i in range(len(self.blocks)):
            remove_weight_norm(self.blocks[i][0])
            self.blocks[i][1].remove_weight_norm()
