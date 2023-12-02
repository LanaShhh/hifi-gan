import itertools

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
from utils import save_checkpoint
from wandb_writer import WanDBWriter
import numpy as np

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

# TODO checkpoint
# TODO wandb

# TODO make folders

# TODO wandb logging through args

cur_step = len(train_dataset) * (train_config.last_epoch + 1)
logger = WanDBWriter(train_config)
tqdm_bar = tqdm(total=(train_config.epochs - train_config.last_epoch - 1) * len(dataloader))

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
        logger.add_scalar('sum_disc_loss', loss_all)

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

        logger.add_scalar('mpd_feat_loss', feature_loss_mpd)
        logger.add_scalar('mpd_gen_loss', loss_pgen)
        logger.add_scalar('msd_feat_loss', feature_loss_msd)
        logger.add_scalar('msd_gen_loss', loss_sgen)
        logger.add_scalar('mel_loss', loss_mel)
        logger.add_scalar('gen_total_loss', loss_all)

        if cur_step > 0 and cur_step % train_config.save_step == 0:
            logger.set_step(cur_step, 'test')
            generator.eval()

            with torch.no_grad():
                err = 0
                for i, wav in enumerate(test_dataset):
                    wav = wav.unsqueeze(0).to(device)
                    mel = melspec(wav)

                    fake_wav = generator(mel)
                    fake_mel = melspec(fake_wav)

                    logger.add_audio(f'genarated_epoch={i}', fake_wav, melspec_config.sr)

                    err += F.l1_loss(mel, fake_mel).item()

                logger.add_scalar('melspec_error', err)

                save_checkpoint(f"{train_config.checkpoint_path}/checkpoint_{cur_step}")

    g_scheduler.step()
    d_scheduler.step()
