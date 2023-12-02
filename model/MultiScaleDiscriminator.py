# based on https://github.com/jik876/hifi-gan

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm

from configs import TrainConfig


class ScaleDiscriminator(nn.Module):
    def __init__(self, train_config:TrainConfig, norm=weight_norm):
        super(ScaleDiscriminator, self).__init__()
        self.relu_coef = train_config.relu

        self.convs = nn.ModuleList([
            norm(nn.Conv1d(1, 128, 15, 1, padding=7)),
            norm(nn.Conv1d(128, 128, 41, 2, padding=20, groups=4)),
            norm(nn.Conv1d(128, 256, 41, 2, padding=20, groups=16)),
            norm(nn.Conv1d(256, 512, 41, 4, padding=20, groups=16)),
            norm(nn.Conv1d(512, 1024, 41, 4, padding=20, groups=16)),
            norm(nn.Conv1d(1024, 1024, 41, 1, padding=20, groups=16)),
            norm(nn.Conv1d(1024, 1024, 5, 1, padding=2))
        ])

        self.post_conv = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1,
                                               padding=(1, 0)))

    def forward(self, x):
        features = []
        out = x

        for conv in self.convs:
            out = conv(out)
            out = F.leaky_relu(out, self.relu_coef)
            features.append(out)

        out = self.post_conv(out)
        features.append(out)
        out = torch.flatten(out, 1, -1)
        return out, features


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, train_config: TrainConfig):
        super(MultiScaleDiscriminator, self).__init__()

        self.scale_discriminators = nn.ModuleList([
            ScaleDiscriminator(train_config, norm=spectral_norm),
            ScaleDiscriminator(train_config),
            ScaleDiscriminator(train_config)
        ])

        self.avg_pools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y_real, y_fake):
        out_real, features_real = [], []
        out_fake, features_fake = [], []

        for i in range(len(self.scale_discriminators)):
            discriminator = self.scale_discriminators[i]
            if i != 0:
                y_real = self.avg_pools[i - 1](y_real)
                y_fake = self.avg_pools[i - 1](y_fake)

            res_real = discriminator(y_real)
            res_fake = discriminator(y_fake)

            out_real.append(res_real[0])
            features_real.append(res_real[1])

            out_fake.append(res_fake[0])
            features_fake.append(res_fake[1])

        return out_real, out_fake, features_real, features_fake
