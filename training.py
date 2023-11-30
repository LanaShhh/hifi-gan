from configs import train_config
from dataset.ModelDataset import ModelDataset
from dataset.download import download_archive
from torch.utils.data import DataLoader

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







