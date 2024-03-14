# import torch
from torch.utils.data import Dataset


class Organoids(Dataset):
    def __init__(self, file="organoids.csv", path="", enable_preprocessing=True):
        # get data csv
        pass

    def __len__(self):
        # number of data samples
        pass

    def __getitem__(self, idx):
        # preprocess and return sample
        pass
