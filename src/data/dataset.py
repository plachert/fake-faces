from torch.utils.data import Dataset
import pathlib
import cv2
import torch


class ThisPersonDoesNotExistDatset(Dataset):
    """Dataset with unique samples from https://thispersondoesnotexist.com"""
    def __init__(self, data_path: pathlib.Path, transform=None) -> None:
        super().__init__()
        self.img_paths = list(data_path.glob("*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img = self._read_image(self.img_paths[idx])
        if self.transform:
            img = self.transform(img)
        return img
    
    def _read_image(self, path: pathlib.Path):
        img_bgr = cv2.imread(str(path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img_rgb)
        img_channel_first = img.permute(2, 0, 1) # channel last to channel first
        return img_channel_first.to(dtype=torch.float32)
        
        
        
    
    
if __name__ == "__main__":
    ds = ThisPersonDoesNotExistDatset(pathlib.Path("/home/piotr/datasets/vision/fake_faces"))
    print(ds[0])