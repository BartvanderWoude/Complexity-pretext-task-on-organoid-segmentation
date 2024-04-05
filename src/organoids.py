import src.pretext_tasks as pt

import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image


class Organoids(Dataset):
    def __init__(self, csv_path, path="./", data_path="organoid_data/", task1="", task2=""):
        self.df = pd.read_csv(path + csv_path, header=None, names=["stacks", "masks"])
        self.path = path + data_path

        self.task1 = task1
        self.task2 = task2

        self.resize = transforms.Compose([transforms.Resize(320, antialias=True),])
        self.transform_task1 = pt.get_distortion_transform(task1)
        self.transform_task2 = pt.get_distortion_transform(task2)
        self.normalize = transforms.Compose([transforms.ConvertImageDtype(torch.float32),])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        assert self.task2 != "j", "Jigsaw must be task 1"
        assert self.task2 != "p", "Predict rotation must be task 1"
        if self.task1 == "j" or self.task1 == "p":
            assert not self.task2, "Jigsaw/predict rotation must be single task only"

        img_path = self.path + self.df.iloc[idx, 0]
        image = read_image(img_path)

        image = self.resize(image)

        if not self.task1 and not self.task2:
            gt_path = self.path + self.df.iloc[idx, 1]
            gt = read_image(gt_path)
            gt = self.resize(gt)
        if self.task1 == "j" or self.task1 == "p":
            image, gt = self.transform_task1(image)
            image = self.normalize(image)
            return (image, gt)
        else:
            gt = image.clone()

        gt = self.normalize(gt)

        image = self.transform_task1(image)
        image = self.transform_task2(image)
        image = self.normalize(image)

        return (image, gt)
