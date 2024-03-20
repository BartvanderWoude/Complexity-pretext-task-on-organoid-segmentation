import torch
import torchvision.transforms as transforms


def get_distortion_transform(task):
    possible_tasks = ["b", "d", "s", "r", "B", "D", "S", "R"]
    assert task in possible_tasks, "Invalid task. Possible tasks: " + str(possible_tasks)

    if task == "b":
        return _blur


def _blur(image):
    transform = transforms.Compose([
            transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
            transforms.Resize(320, antialias=True),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    return transform(image)
