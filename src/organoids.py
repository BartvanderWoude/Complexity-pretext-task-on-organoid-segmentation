import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image

import pandas as pd


class Organoids(Dataset):
    def __init__(self, file, path="./", transform=None):
        self.path = path + "organoid_data/"
        self.df = pd.read_csv(path + file, header=None, names=["path"])
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(320, antialias=True),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.path + self.df.iloc[idx, 0]
        image = read_image(img_path)
        image = self.transform(image)

        return image
