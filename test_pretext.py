import torch
import argparse
import os
from torch.utils.data import DataLoader

import src.model as m
import src.organoids as org
import src.logger as lg
import src.evaluation as ev


def get_pretrained_model(task1="", task2=""):
    path = f"output/models/pretext/{task1}_{task2}/"
    model_name = ""
    highest_fold = -1
    for file in os.listdir(path):
        file_components = file.split("_")
        if int(file_components[0][1:]) > highest_fold:
            model_name = file
            highest_fold = int(file_components[0][1:])

    print(f"Loading model: {model_name}")

    return path + model_name, highest_fold


def test_pretext(task1, task2):
    batch_size = 16

    print(f"Pretext testing with tasks: {task1}, {task2}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = m.UNet(1, 1)
    logger = lg.Logger("pretext", task1, task2, evaluating=True)

    # Load pretext trained model
    pretrained, fold = get_pretrained_model(task1=task1, task2=task2)
    print(f"Pretrained model: {pretrained}")
    model.load_state_dict(torch.load(pretrained))
    model = model.to(device)

    dataset = org.Organoids(file="utils/test.csv", task1=task1, task2=task2)

    print("Fold: ", fold)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    ev.evaluate_selfprediction(task1, task2, fold, model, test_loader, device, logger)


def get_args():
    parser = argparse.ArgumentParser(description='Pretext Testing')
    parser.add_argument('--task1', type=str, default="", help='Specify distortion type for task 1')
    parser.add_argument('--task2', type=str, default="", help='Specify distortion type for task 2')

    args = parser.parse_args()

    return args.task1, args.task2


if __name__ == '__main__':
    task1, task2 = get_args()

    test_pretext(task1=task1, task2=task2)
