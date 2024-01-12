import pathlib
import random
from typing import List, Tuple

import lightning as L
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from src.data.dataset import ThisPersonDoesNotExistDatset


def random_split(list_of_items: List, split: Tuple[float]) -> List[List]:
    assert sum(split) == 1
    shuffled_list = list_of_items.copy()
    random.shuffle(shuffled_list)
    total_size = len(shuffled_list)
    sizes = [int(total_size * ratio) for ratio in split]
    splits = [
        shuffled_list[sum(sizes[:i]) : sum(sizes[: i + 1])] for i in range(len(sizes))
    ]
    return splits


class ThisPersonDoesNotExistDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: pathlib.Path,
        batch_size: int = 64,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = transforms.Resize(size=(500, 500), antialias=True)
        self.val_transform = transforms.Resize(size=(500, 500), antialias=True)

    def setup(self, stage=None) -> None:
        files = list(self.data_dir.glob("*.jpg"))
        train_list, val_list, test_list = random_split(files, (0.9, 0.05, 0.05))
        if stage == "fit" or stage is None:
            self.train_set = ThisPersonDoesNotExistDatset(
                train_list, transform=self.train_transform
            )
            self.val_set = ThisPersonDoesNotExistDatset(
                val_list, transform=self.val_transform
            )
        else:
            self.test_set = ThisPersonDoesNotExistDatset(
                test_list, transform=self.val_transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, batch_size=self.batch_size, num_workers=self.num_workers
        )
