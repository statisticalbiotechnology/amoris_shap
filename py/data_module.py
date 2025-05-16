import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl

class AgeDataset(Dataset):
    def __init__(self, features, targets):
        self.x = torch.tensor(features.values, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class AgeDataModule(pl.LightningDataModule):
    def __init__(self, features, targets, batch_size=64):
        super().__init__()
        self.features = features
        self.targets = targets
        self.batch_size = batch_size

    def setup(self, stage=None):
        dataset = AgeDataset(self.features, self.targets)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset, [14000, len(dataset) - 14000])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
