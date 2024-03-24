import torch

import src.organoids as organoids


def test_organoids_dataset_downstream_train():
    dataset = organoids.Organoids(file="utils/downstream_train.csv")
    sample, gt = dataset[0]

    assert len(dataset) == 40636
    assert sample.shape == (1, 320, 320)
    assert torch.max(sample) <= 1.0
    assert torch.min(sample) >= -1.0


def test_organoids_dataset_pretext_train():
    dataset = organoids.Organoids(file="utils/pretext_train.csv")
    sample, gt = dataset[0]

    assert len(dataset) == 40631
    assert sample.shape == (1, 320, 320)
    assert torch.max(sample) <= 1.0
    assert torch.min(sample) >= -1.0


def test_organoids_dataset_test():
    dataset = organoids.Organoids(file="utils/test.csv")
    sample, gt = dataset[0]

    assert len(dataset) == 20322
    assert sample.shape == (1, 320, 320)
    assert torch.max(sample) <= 1.0
    assert torch.min(sample) >= -1.0
