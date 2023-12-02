import torch
import torch.nn.functional as F


def count_feature_loss(features_real, features_fake, alpha=2):
    loss = 0

    for f_real, f_fake in zip(features_real, features_fake):
        for f1, f2 in zip(f_real, f_fake):
            loss += F.l1_loss(f1, f2)

    return loss * alpha


def count_discriminator_loss(d_out_real, d_out_fake):
    loss = 0
    real_losses = []
    fake_losses = []
    for d_real, d_fake in zip(d_out_real, d_out_fake):
        real_losses.append(F.mse_loss(d_real, torch.full(d_real.size(), 1)).item())
        fake_losses.append(torch.mean(d_fake ** 2).item())
        loss += real_losses[-1] + fake_losses[-1]

    return loss, real_losses, fake_losses


def count_generator_loss(d_out):
    loss = 0
    fake_losses = []
    for d in d_out:
        fake_losses.append(F.mse_loss(d, torch.full(d.size(), 1)).item())
        loss += fake_losses[-1]
    return loss, fake_losses
