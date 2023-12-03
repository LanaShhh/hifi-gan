import argparse
import itertools
import os

import numpy as np
import torchaudio
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import train_config, melspec_config
from dataset.ModelDataset import ModelDataset
from dataset.download import download_archive
from dataset.mel_spec_generation import MelSpectrogram
from loss import *
from model.Generator import Generator
from model.MultiPeriodDiscriminator import MultiPeriodDiscriminator as MPD
from model.MultiScaleDiscriminator import MultiScaleDiscriminator as MSD
from wandb_writer import WanDBWriter

parser = argparse.ArgumentParser(prog="Hifi-GAN training")

parser.add_argument('wandb_key', type=str,
                    help='Wandb key for logging')

args = parser.parse_args()

wandb.login("never", args.wandb_key)

torch.cuda.manual_seed(train_config.seed)
torch.manual_seed(train_config.seed)
np.random.seed(train_config.seed)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

download_archive()

train_dataset = ModelDataset(train_config)
test_dataset = ModelDataset(train_config, crop=True)
dataloader = DataLoader(
    train_dataset,
    batch_size=train_config.batch_size,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
    num_workers=0
)

device = train_config.device

generator = Generator(train_config).to(device)
msd = MSD(train_config).to(device)
mpd = MPD(train_config).to(device)

g_optimizer = torch.optim.AdamW(
    [
        {
            'params': generator.parameters(),
            'initial_lr': train_config.learning_rate * train_config.lr_decay ** (train_config.last_epoch + 1)
        }
    ],
    train_config.learning_rate, betas=(train_config.adam_b1, train_config.adam_b2)
)

d_optimizer = torch.optim.AdamW(
    [
        {
            'params': itertools.chain(msd.parameters(), mpd.parameters()),
            'initial_lr': train_config.learning_rate * train_config.lr_decay ** (train_config.last_epoch + 1)
        }
    ],
    train_config.learning_rate, betas=(train_config.adam_b1, train_config.adam_b2)
)

g_scheduler = torch.optim.lr_scheduler.ExponentialLR(
    g_optimizer,
    gamma=train_config.lr_decay,
    last_epoch=train_config.last_epoch
)

d_scheduler = torch.optim.lr_scheduler.ExponentialLR(
    d_optimizer,
    gamma=train_config.lr_decay,
    last_epoch=train_config.last_epoch
)

melspec = MelSpectrogram(melspec_config).to(device)

cur_step = len(train_dataset) * (train_config.last_epoch + 1)
logger = WanDBWriter(train_config)
tqdm_bar = tqdm(total=(train_config.epochs - train_config.last_epoch - 1) * len(dataloader))

test_audios_dir = {}
for epoch in range(train_config.last_epoch + 1, train_config.epochs):
    for idx, batch in enumerate(dataloader):
        cur_step += 1
        tqdm_bar.update(1)
        logger.set_step(cur_step)

        wav = batch.to(device)
        mel = melspec(wav)

        generator.train()
        mpd.train()
        msd.train()

        fake_wav = generator(mel)
        fake_mel = melspec(fake_wav)

        d_optimizer.zero_grad()
        real_res, fake_res, _, _ = mpd(wav, fake_wav.detach())
        loss_mpd = count_discriminator_loss(real_res, fake_res)

        real_res, fake_res, _, _ = msd(wav, fake_wav.detach())
        loss_msd = count_discriminator_loss(real_res, fake_res)

        loss_all = loss_mpd + loss_msd
        loss_all.backward()
        d_optimizer.step()

        logger.add_scalar('mpd_loss', loss_mpd)
        logger.add_scalar('msd_loss', loss_msd)
        logger.add_scalar('discriminator_loss', loss_all)

        g_optimizer.zero_grad()

        _, fake_res, features_real, features_fake = mpd(wav, fake_wav)
        feature_loss_mpd = count_feature_loss(features_real, features_fake)
        loss_pgen = count_generator_loss(fake_res)

        _, fake_res, features_real, features_fake = msd(wav, fake_wav)
        feature_loss_msd = count_feature_loss(features_real, features_fake)
        loss_sgen = count_generator_loss(fake_res)

        loss_mel = F.l1_loss(mel, fake_mel) * 45
        loss_all = loss_pgen + feature_loss_mpd + loss_sgen + feature_loss_msd + loss_mel
        loss_all.backward()
        g_optimizer.step()

        logger.add_scalar('mpd_feature_loss', feature_loss_mpd)
        logger.add_scalar('mpd_gen_loss', loss_pgen)
        logger.add_scalar('msd_feat_loss', feature_loss_msd)
        logger.add_scalar('msd_gen_loss', loss_sgen)
        logger.add_scalar('mel_loss', loss_mel)
        logger.add_scalar('generator_loss', loss_all)

        if cur_step > 0 and cur_step % train_config.save_step == 0:
            generator.eval()
            mpd.eval()
            msd.eval()

            with torch.no_grad():
                err = 0
                for i, wav in enumerate(test_dataset):
                    if i >= 500:
                        break
                    wav = wav.unsqueeze(0).to(device)
                    mel = melspec(wav)

                    fake_wav = generator(mel)
                    fake_mel = melspec(fake_wav)

                    err += F.l1_loss(mel, fake_mel).item()

            if not os.path.exists(train_config.checkpoint_audio_path):
                os.makedirs(train_config.checkpoint_audio_path, exist_ok=True)

            wav = torchaudio.load(os.path.join(train_config.test_audio_path, f"audio_1.wav"))[0][0].to(device)
            mel = melspec(wav)
            fake_wav = generator(mel)
            fake_mel = melspec(fake_wav)
            new_id = len(test_audios_dir.keys())
            torchaudio.save(
                f"{train_config.checkpoint_audio_path}/audio_step_{cur_step}.wav",
                fake_wav.cpu(), melspec_config.sr
            )
            wav = logger.wandb.Audio(os.path.join(train_config.test_audio_path, f"audio_1.wav"), sample_rate=melspec_config.sr)
            fake_wav = logger.wandb.Audio(f"{train_config.checkpoint_audio_path}/audio_step_{cur_step}.wav", sample_rate=melspec_config.sr)
            test_audios_dir[new_id] = {"cur_step": cur_step, "real_wav": wav,
                                       "generated_wav": fake_wav}
            logger.add_table(test_audios_dir)
            logger.add_scalar('test_melspec_error', err)

            if not os.path.exists(train_config.checkpoint_path):
                os.makedirs(train_config.checkpoint_path, exist_ok=True)

            torch.save(
                {
                    'step': cur_step,
                    'generator': generator.state_dict(),
                    'mpd': mpd.state_dict(),
                    'msd': msd.state_dict(),
                    'd_optimizer': d_optimizer.state_dict(),
                    'g_optimizer': g_optimizer.state_dict()
                },
                os.path.join(train_config.checkpoint_path, 'checkpoint.pth.tar'))
            print("save generator at step %d ..." % cur_step)

    g_scheduler.step()
    d_scheduler.step()
