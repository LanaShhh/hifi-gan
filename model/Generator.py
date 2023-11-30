import torch.functional as F
from torch import nn
from torch.nn.utils import weight_norm

from configs import TrainConfig


class ResBlock(nn.Module):
    def __init__(self, train_config: TrainConfig, channels, kernel_size=3):
        super(ResBlock, self).__init__()

        self.convs = nn.ModuleList([])
        self.relu_ratio = train_config.relu

        for stride, dilation in train_config.D_r:
            new_convs = nn.ModuleList([
                weight_norm(nn.Conv1d(channels, channels, kernel_size,
                                      stride=stride, dilation=dilation,
                                      padding=int((kernel_size * dilation - dilation) / 2))),
                weight_norm(nn.Conv1d(channels, channels, kernel_size,
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
