import pathlib
from typing import List

from PIL import Image
from torch.utils.data import Dataset


class ThisPersonDoesNotExistDatset(Dataset):
    """Dataset with unique samples from https://thispersondoesnotexist.com"""

    def __init__(self, img_paths: List[pathlib.Path], transform=None) -> None:
        super().__init__()
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = self._read_image(self.img_paths[idx])
        if self.transform:
            img = self.transform(img)
        return img

    def _read_image(self, path: pathlib.Path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")
