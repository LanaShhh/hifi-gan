# based on https://github.com/jik876/hifi-gan

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from configs import TrainConfig


class PeriodDiscriminator(nn.Module):
    def __init__(self, train_config: TrainConfig, period, kernel_size=5, stride=3):
        super(PeriodDiscriminator, self).__init__()
        self.period = period
        self.relu_coef = train_config.relu

        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1),
                                  padding=(2, 0))),
            weight_norm(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1),
                                  padding=(2, 0))),
            weight_norm(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1),
                                  padding=(2, 0))),
            weight_norm(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1),
                                  padding=(2, 0))),
            weight_norm(nn.Conv2d(1024, 1024, (kernel_size, 1), (stride, 1),
                                  padding=(2, 0)))
        ])

        self.post_conv = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1,
                                               padding=(1, 0)))

    def forward(self, x):
        features = []
        batch, signal, time = x.shape
        if time % self.period != 0:
            x = F.pad(x, (0, self.period - (time % self.period)), "reflect")
            time += self.period - (time % self.period)
        out = x.view(batch, signal, time // self.period, self.period)

        for conv in self.convs:
            out = conv(out)
            out = F.leaky_relu(out, self.relu_coef)
            features.append(out)

        out = self.post_conv(out)
        features.append(out)
        out = torch.flatten(out, 1, -1)
        return out, features


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, train_config: TrainConfig):
        super(MultiPeriodDiscriminator, self).__init__()
        self.period_discriminators = nn.ModuleList([
            PeriodDiscriminator(train_config, 2),
            PeriodDiscriminator(train_config, 3),
            PeriodDiscriminator(train_config, 5),
            PeriodDiscriminator(train_config, 7),
            PeriodDiscriminator(train_config, 11)
        ])

    def forward(self, y_real, y_fake):
        out_real, features_real = [], []
        out_fake, features_fake = [], []

        for discriminator in self.period_discriminators:
            res_real = discriminator(y_real)
            res_fake = discriminator(y_fake)

            out_real.append(res_real[0])
            features_real.append(res_real[1])

            out_fake.append(res_fake[0])
            features_fake.append(res_fake[1])

        return out_real, out_fake, features_real, features_fake
