import os

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

from configs import TrainConfig


class ModelDataset(Dataset):
    def __init__(self, train_config: TrainConfig, crop=True):
        self.wav_dir_path = train_config.wav_path
        self.wav_names_list = sorted(os.listdir(self.wav_dir_path))
        self.crop = crop
        self.crop_len = train_config.crop_len if crop else None

    def __len__(self):
        return len(self.wav_names_list)

    def __getitem__(self, index):
        if index >= len(self.wav_names_list):
            raise Exception("Index is too big")
        wav = torchaudio.load(os.path.join(self.wav_dir_path, self.wav_names_list[index]))[0][0]
        if self.crop:
            if wav.size(-1) < self.crop_len:
                wav = F.pad(wav, (0, self.crop_len - wav.size(-1)))
            else:
                start_id = torch.randint(0, wav.size(-1) - self.crop_len + 1, (1, )).item()
                wav = wav[start_id:start_id + self.crop_len]
        return wav
