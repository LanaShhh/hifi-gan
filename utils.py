import torch

def save_checkpoint(checkpoint_path, generator, mpd, msp, g_optimizer, d_optimizer):
    checkpoint = torch.load(checkpoint_path)


