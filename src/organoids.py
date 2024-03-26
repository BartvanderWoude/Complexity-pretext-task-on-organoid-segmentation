import src.selfprediction_distortion as spd

import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image


class Organoids(Dataset):
    def __init__(self, file, path="./", task1="", task2=""):
        self.df = pd.read_csv(path + file, header=None, names=["stacks", "masks"])
        self.path = path + "organoid_data/"

        self.task1 = task1
        self.task2 = task2

        self.prepare_transform = transforms.Compose([
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
            transforms.Resize(320, antialias=True),
        ])
        self.transform_task1 = spd.get_distortion_transform(task1)
        self.transform_task2 = spd.get_distortion_transform(task2)
        self.basic_transform = transforms.Compose([
            transforms.ConvertImageDtype(torch.float32),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.path + self.df.iloc[idx, 0]
        image = read_image(img_path)

        image = self.prepare_transform(image)

        if not self.task1 and not self.task2:
            gt_path = self.path + self.df.iloc[idx, 1]
            gt = read_image(gt_path)

            gt = self.prepare_transform(gt)
            gt = self.basic_transform(gt)

            image = self.basic_transform(image)

            return (image, gt)

        gt = image.clone()
        gt = self.basic_transform(gt)

        image = self.transform_task1(image)
        image = self.transform_task2(image)
        image = self.basic_transform(image)

        return (image, gt)
