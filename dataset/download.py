import logging
import os
import shutil
from pathlib import Path

from speechbrain.utils.data_utils import download_file

from configs import train_config


def download_archive():
    if not Path(train_config.data_path).exists():
        os.makedirs(train_config.data_path, exist_ok=True)
    if not Path(train_config.dataset_archive_save_path).exists():
        print(f"Downloading and unpacking dataset archive: {train_config.dataset_archive_path}")
        download_file(train_config.dataset_archive_path, dest=train_config.dataset_archive_save_path,
                      unpack=True, dest_unpack=train_config.data_path)
