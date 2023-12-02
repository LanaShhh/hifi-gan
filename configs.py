from dataclasses import dataclass

import torch

@dataclass
class MelSpectrogramConfig:
    sr: int = 22050
    win_length: int = 1024
    hop_length: int = 256
    n_fft: int = 1024
    f_min: int = 0
    f_max: int = 8000
    n_mels: int = 80
    power: float = 1.0

    # value of melspectrograms if we fed a silence into `MelSpectrogram`
    pad_value: float = -11.5129251


@dataclass
class TrainConfig:
    checkpoint_path = "./checkpoints"
    logger_path = "./logger"

    data_path = "./data"
    dataset_folder = "./data/LJSpeech-1.1"
    wav_path = "./data/LJSpeech-1.1/wavs"
    dataset_archive_path = 'https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2'
    dataset_archive_save_path = './data/LJSpeech-1.1.tar.bz2'

    test_audio_path = "./test_audios"
    inf_audio_path = "./results"
    checkpoint_audio_path = "./checkpoints_audios"

    wandb_project = 'hifigan_sdzhumlyakova_implementation'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    last_epoch = -1
    save_step = 10
    epochs = 30

    # from original model config
    crop_len = 8192
    batch_size = 16

    D_r = [[1, 1], [3, 1], [5, 1]]

    relu = 0.1

    upsample_rates = [8, 8, 2, 2]
    upsample_kernel_sizes = [16, 16, 4, 4]
    upsample_initial_channel = 512
    resblock_kernel_sizes = [3, 7, 11]

    learning_rate = 0.0002
    adam_b1 = 0.8
    adam_b2 = 0.99
    lr_decay = 0.999
    seed = 1234

    # for inference
    # texts to generate
    texts = [
        "A defibrillator is a device that gives a high energy electric shock to the heart of someone "
        "who is in cardiac arrest",
        "Massachusetts Institute of Technology may be best known for its math, science and engineering education",
        "Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined "
        "between probability distributions on a given metric space"
    ]

    # for training
    # text logged audios
    logging_text = ("A defibrillator is a device that gives a high energy electric shock to the heart of someone "
                    "who is in cardiac arrest")


train_config = TrainConfig()
melspec_config = MelSpectrogramConfig()
