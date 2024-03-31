import torch
import torchvision.transforms as transforms
import math


def get_distortion_transform(task):
    possible_tasks = ["", "b", "d", "s", "r", "B", "D", "S", "R", "j", "p"]
    assert task in possible_tasks, "Invalid task" + str(task) + ". Possible tasks: " + str(possible_tasks)

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
    elif task == "B":
        return _blur_boxes
    elif task == "D":
        return _drop_pixel
    elif task == "S":
        return _shuffle_rotate
    elif task == "R":
        return _rotate_boxes
    elif task == "j":
        return _jigsaw
    elif task == "p":
        return _predict_rotation


def _do_nothing(image):
    return image


def _blur(image):
    return transforms.functional.gaussian_blur(image, 5, sigma=(2, 3))


def _drop(image):
    random_boxes = torch.randint(320 - 50, (4, 2))

    for b in range(4):
        (x1, x2, y1, y2) = (random_boxes[b, 0],
                            random_boxes[b, 0] + 50,
                            random_boxes[b, 1],
                            random_boxes[b, 1] + 50)
        image[:, y1:y2, x1:x2] = 0

    return image


def _shuffle(image):
    random_boxes = torch.randint(320 - 50, (4, 2))
    boxes = torch.zeros(4, 50, 50)

    for b in range(4):
        (x1, x2, y1, y2) = (random_boxes[b, 0],
                            random_boxes[b, 0] + 50,
                            random_boxes[b, 1],
                            random_boxes[b, 1] + 50)
        boxes[b, :, :] = image[:, y1:y2, x1:x2]

    for b in range(4):
        (x1, x2, y1, y2) = (random_boxes[b, 0],
                            random_boxes[b, 0] + 50,
                            random_boxes[b, 1],
                            random_boxes[b, 1] + 50)
        image[:, y1:y2, x1:x2] = boxes[(b+1) % 4, :, :]

    return image


def _rotate(image):
    boxes = torch.zeros(4, 48, 48)
    random_boxes = torch.randint(320 - 48, (4, 2))
    positions = torch.zeros(4, 4)
    for b in range(4):
        (x1, x2, y1, y2) = (random_boxes[b, 0],
                            random_boxes[b, 0] + 48,
                            random_boxes[b, 1],
                            random_boxes[b, 1] + 48)
        positions[b, :] = torch.tensor([x1, x2, y1, y2])
        boxes[b, :, :] = image[:, y1:y2, x1:x2]

    angles = torch.randint(359, (4,))

    for b in range(4):
        box = boxes[b, :, :].unsqueeze(0)
        box = transforms.functional.rotate(box, int(angles[b]))
        box = torch.clamp(box.squeeze(0), max=255)

        for i in range(48):
            for j in range(48):
                if math.sqrt((i-23)**2 + (j-23)**2) <= 24:
                    x = int(positions[b, 0]) + i
                    y = int(positions[b, 2]) + j

                    image[:, y, x] = box[j, i]

    return image


def _blur_boxes(image):
    random_boxes = torch.randint(320 - 50, (4, 2))

    for b in range(4):
        (x1, x2, y1, y2) = (random_boxes[b, 0],
                            random_boxes[b, 0] + 50,
                            random_boxes[b, 1],
                            random_boxes[b, 1] + 50)
        image[:, y1:y2, x1:x2] = transforms.functional.gaussian_blur(image[:, y1:y2, x1:x2], 5, sigma=16)

    return image


def _drop_pixel(image):
    random_boxes = torch.randint(320 - 50, (4, 2))
    drop_pixel = torch.randint(100, (4, 50, 50))

    for b in range(4):
        (x1, y1) = (random_boxes[b, 0],
                    random_boxes[b, 1])

        for i in range(50):
            for j in range(50):
                if drop_pixel[b, i, j] > 50:
                    image[:, y1+j, x1+i] = 0

    return image


def _shuffle_rotate(image):
    boxes = torch.zeros(4, 48, 48)
    random_boxes = torch.randint(320 - 48, (4, 2))

    for b in range(4):
        (x1, x2, y1, y2) = (random_boxes[b, 0],
                            random_boxes[b, 0] + 48,
                            random_boxes[b, 1],
                            random_boxes[b, 1] + 48)
        boxes[b, :, :] = image[:, y1:y2, x1:x2]

    angles = torch.randint(359, (4,))

    for b in range(4):
        box = boxes[b, :, :].unsqueeze(0)
        box = transforms.functional.rotate(box, int(angles[b]))
        box = torch.clamp(box.squeeze(0), max=255)

        for i in range(48):
            for j in range(48):
                if math.sqrt((i-23)**2 + (j-23)**2) <= 24:
                    x = int(random_boxes[(b+1) % 4, 0]) + i
                    y = int(random_boxes[(b+1) % 4, 1]) + j

                    image[:, y, x] = box[j, i]

    return image


def _rotate_boxes(image):
    random_boxes = torch.randint(320 - 50, (4, 2))
    random_rotations = torch.randint(1, 3, (4,))

    for b in range(4):
        (x1, x2, y1, y2) = (random_boxes[b, 0],
                            random_boxes[b, 0] + 50,
                            random_boxes[b, 1],
                            random_boxes[b, 1] + 50)
        box = torch.squeeze(image[:, y1:y2, x1:x2])
        box = torch.rot90(box, random_rotations[b])
        image[:, y1:y2, x1:x2] = box[:, :]

    return image


def _jigsaw(image):
    raster = torch.zeros(9, 106, 106)
    for i in range(3):
        for j in range(3):
            raster[i*3+j, :, :] = image[:, i*106:(i+1)*106, j*106:(j+1)*106]
    indices = torch.randperm(9)
    for i in range(3):
        for j in range(3):
            image[:, i*106:(i+1)*106, j*106:(j+1)*106] = raster[indices[i*3+j], :, :]

    return image, indices


def _predict_rotation(image):
    center = torch.zeros(150, 150)
    center[:, :] = image[:, 85:235, 85:235]

    angle = torch.randint(3, (1,))
    center = torch.rot90(center, angle.item())
    image[:, 85:235, 85:235] = center[:, :]

    return image, angle
