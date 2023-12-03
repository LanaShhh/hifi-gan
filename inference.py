import argparse
import os.path

import torch
import torchaudio

from configs import train_config, melspec_config
from dataset.mel_spec_generation import MelSpectrogram
from model.Generator import Generator

parser = argparse.ArgumentParser(prog="HifiGAN inference")

parser.add_argument('model_state_dict_path', type=str,
                    help='relative path for model state dict')

args = parser.parse_args()

generator = Generator(train_config)
generator.load_state_dict(torch.load(args.model_state_dict_path, map_location='cuda:0')['generator'])
generator = generator.to(train_config.device)
generator = generator.eval()

melspec = MelSpectrogram(melspec_config).to(train_config.device)

if not os.path.exists(train_config.inf_audio_path):
    os.makedirs(train_config.inf_audio_path)

for i in [1, 2, 3]:
    wav = (torchaudio.load(os.path.join(train_config.test_audio_path, f"audio_{i}.wav"))[0][0]
           .to(train_config.device))
    mel = melspec(wav)
    fake_wav = generator(mel)
    fake_mel = melspec(fake_wav)
    torchaudio.save(
        f"{train_config.inf_audio_path}/audio_{i}.wav",
        fake_wav.cpu(), melspec_config.sr
    )
