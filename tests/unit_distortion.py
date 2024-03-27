import torch

import src.organoids as organoids


def sample_check(sample):
    assert sample.shape == (1, 320, 320)
    assert torch.max(sample) <= 1.0
    assert torch.min(sample) >= 0.0


def gt_check(gt):
    assert gt.shape == (1, 320, 320)
    assert torch.max(gt) <= 1.0
    assert torch.min(gt) >= 0.0


def test_selfprediction_distortion_blur():
    dataset = organoids.Organoids(file="utils/pretext_train.csv", task1="b", task2="")
    sample, gt = dataset[0]

    sample_check(sample)
    gt_check(gt)


def test_selfprediction_distortion_drop():
    dataset = organoids.Organoids(file="utils/pretext_train.csv", task1="d", task2="")
    sample, gt = dataset[0]

    sample_check(sample)
    gt_check(gt)


def test_selfprediction_distortion_shuffle():
    dataset = organoids.Organoids(file="utils/pretext_train.csv", task1="s", task2="")
    sample, gt = dataset[0]

    sample_check(sample)
    gt_check(gt)


def test_selfprediction_distortion_rotate():
    dataset = organoids.Organoids(file="utils/pretext_train.csv", task1="r", task2="")
    sample, gt = dataset[0]

    sample_check(sample)
    gt_check(gt)


def test_selfprediction_distortion_blur_boxes():
    dataset = organoids.Organoids(file="utils/pretext_train.csv", task1="B", task2="")
    sample, gt = dataset[0]

    sample_check(sample)
    gt_check(gt)


def test_selfprediction_distortion_drop_pixel():
    dataset = organoids.Organoids(file="utils/pretext_train.csv", task1="D", task2="")
    sample, gt = dataset[0]

    sample_check(sample)
    gt_check(gt)


def test_selfprediction_distortion_shuffle_rotate():
    dataset = organoids.Organoids(file="utils/pretext_train.csv", task1="S", task2="")
    sample, gt = dataset[0]

    sample_check(sample)
    gt_check(gt)


def test_selfprediction_distortion_rotate_boxes():
    dataset = organoids.Organoids(file="utils/pretext_train.csv", task1="R", task2="")
    sample, gt = dataset[0]

    sample_check(sample)
    gt_check(gt)


def test_selfprediction_distortion_combination():
    dataset = organoids.Organoids(file="utils/pretext_train.csv", task1="R", task2="B")
    sample, gt = dataset[0]

    sample_check(sample)
    gt_check(gt)


def test_selfprediction_distortion_combination2():
    dataset = organoids.Organoids(file="utils/pretext_train.csv", task1="S", task2="r")
    sample, gt = dataset[0]

    sample_check(sample)
    gt_check(gt)


def test_selfprediction_distortion_combination3():
    dataset = organoids.Organoids(file="utils/pretext_train.csv", task1="D", task2="b")
    sample, gt = dataset[0]

    sample_check(sample)
    gt_check(gt)


def test_selfprediction_distortion_combination4():
    dataset = organoids.Organoids(file="utils/pretext_train.csv", task1="s", task2="d")
    sample, gt = dataset[0]

    sample_check(sample)
    gt_check(gt)


def test_selfprediction_distortion_no_task():
    dataset = organoids.Organoids(file="utils/pretext_train.csv", task1="", task2="")
    sample, gt = dataset[0]

    sample_check(sample)
    gt_check(gt)
