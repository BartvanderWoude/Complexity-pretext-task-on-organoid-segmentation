import torch
import torchvision.transforms as transforms
import random as rd
import math


def get_distortion_transform(task):
    possible_tasks = ["", "b", "d", "s", "r", "B", "D", "S", "R"]
    assert task in possible_tasks, "Invalid task. Possible tasks: " + str(possible_tasks)

    if task == "":
        return _do_nothing
    elif task == "b":
        return _blur
    elif task == "d":
        return _drop
    elif task == "s":
        return _shuffle
    elif task == "r":
        return _rotate


def _do_nothing(image):
    return image


def _blur(image):
    transform = transforms.Compose([
            transforms.Resize(320, antialias=True),
            transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
        ])

    return transform(image)


def _drop(image):
    transform = transforms.Compose([
            transforms.Resize(320, antialias=True),
        ])
    image = transform(image)

    random_boxes = torch.randint(320 - 50, (4, 2))

    for b in range(4):
        (x1, x2, y1, y2) = (random_boxes[b, 0],
                            random_boxes[b, 0] + 50,
                            random_boxes[b, 1],
                            random_boxes[b, 1] + 50)
        image[:, y1:y2, x1:x2] = 0

    return image


def _shuffle(image):
    transform = transforms.Compose([
            transforms.Resize(320, antialias=True),
        ])
    image = transform(image)

    boxes = torch.zeros(4, 50, 50)
    random_boxes = torch.randint(320 - 50, (4, 2))
    positions = torch.zeros(4, 4)
    for i in range(4):
        (x1, x2, y1, y2) = (random_boxes[i, 0],
                            random_boxes[i, 0] + 50,
                            random_boxes[i, 1],
                            random_boxes[i, 1] + 50)
        positions[i, :] = torch.tensor([x1, x2, y1, y2])
        boxes[i, :, :] = image[:, y1:y2, x1:x2]

    for i in range(4):
        (x1, x2, y1, y2) = (int(positions[i, 0].item()),
                            int(positions[i, 1].item()),
                            int(positions[i, 2].item()),
                            int(positions[i, 3].item()))
        image[:, y1:y2, x1:x2] = boxes[(i+1) % 4, :, :]

    return image


def _rotate(image):
    transform = transforms.Compose([
            transforms.Resize(320, antialias=True),
        ])
    image = transform(image)

    boxes = torch.zeros(4, 50, 50)
    random_boxes = torch.randint(320 - 50, (4, 2))
    positions = torch.zeros(4, 4)
    for b in range(4):
        (x1, x2, y1, y2) = (random_boxes[b, 0],
                            random_boxes[b, 0] + 50,
                            random_boxes[b, 1],
                            random_boxes[b, 1] + 50)
        positions[b, :] = torch.tensor([x1, x2, y1, y2])
        boxes[b, :, :] = image[:, y1:y2, x1:x2]

    boxes = transforms.functional.resize(boxes, 80, antialias=True)
    angles = torch.randint(360, (4,))

    for b in range(4):
        box = boxes[b, :, :].unsqueeze(0)
        box = transforms.functional.rotate(box, int(angles[b]))
        box = transforms.functional.resize(box, 50, antialias=True)
        box = torch.clamp(box.squeeze(0), max=255)

        for i in range(50):
            for j in range(50):
                if math.sqrt((i-24)**2 + (j-24)**2) <= 25:
                    x = int(positions[b, 0].item()) + i
                    y = int(positions[b, 2].item()) + j

                    image[:, y, x] = box[j, i]

    return image
