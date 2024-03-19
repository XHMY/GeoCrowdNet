import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from helpers.data_load import *
import logging


class CrowdsourcingDataModule(pl.LightningDataModule):
    def __init__(self, dataset_name, data_dir, batch_size, num_workers, train_split=0.8, val_split=0.1, test_split=0.1,
                 seed=42, **kwargs):
        super().__init__()
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        self.kwargs = kwargs
        self.dataset_class = self._get_dataset_class()

    def _get_dataset_class(self):
        if self.dataset_name == "mnist":
            return mnist_dataset
        elif self.dataset_name in ["cifar10", "food11"]:
            return cifar10_dataset
        elif self.dataset_name == "labelme":
            return labelme_dataset
        elif self.dataset_name == "music":
            return music_dataset
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.full_dataset = self.dataset_class(train=True, **self.kwargs)
            train_size = int(len(self.full_dataset) * self.train_split)
            val_size = int(len(self.full_dataset) * self.val_split)
            test_size = len(self.full_dataset) - train_size - val_size

            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                self.full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(self.seed)
            )
        if stage == 'test' or stage is None:
            self.test_dataset = self.dataset_class(train=False, **self.kwargs)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
                          pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=True, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=0,
                          pin_memory=True, persistent_workers=False)


if __name__ == '__main__':
    from main import parse_args
    # test MusicDataModule
    dm = CrowdsourcingDataModule(dataset_name="music", data_dir="data", logger=logging.getLogger(),
                                 batch_size=32, num_workers=4, args=parse_args())
    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()
    for batch in train_loader:
        for value in batch:
            print(value.shape)
        break
