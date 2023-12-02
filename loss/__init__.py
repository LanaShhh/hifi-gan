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
    for d_real, d_fake in zip(d_out_real, d_out_fake):
        real_loss = F.mse_loss(d_real.float(), torch.ones_like(d_real).float())
        fake_loss = F.mse_loss(d_fake.float(), torch.zeros_like(d_fake).float())
        loss += real_loss + fake_loss

    return loss


def count_generator_loss(d_out):
    loss = 0
    for d in d_out:
        fake_loss = F.mse_loss(d.float(), torch.ones_like(d).float())
        loss += fake_loss
    return loss
