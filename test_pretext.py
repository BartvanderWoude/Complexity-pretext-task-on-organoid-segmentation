import src.model as m
import src.organoids as org
import src.logger as lg
import src.evaluation as ev

import torch
import argparse
import os
from torch.utils.data import DataLoader


def get_args():
    parser = argparse.ArgumentParser(description='Pretext Testing')
    parser.add_argument('--task1', type=str, default="", help='Specify distortion type for task 1')
    parser.add_argument('--task2', type=str, default="", help='Specify distortion type for task 2')

    args = parser.parse_args()

    return args.task1, args.task2


def get_pretrained_model(task1="", task2=""):
    model_path = f"output/models/pretext/{task1}_{task2}/"

    if not os.path.exists(model_path):
        raise ValueError("No pretext models found.")

    models = os.listdir(model_path)
    models = sorted(models)
    model_paths = [model_path + model for model in models]

    print(f"Loading models: {model_paths}")

    return model_paths


def test_pretext(task1, task2):
    batch_size = 16

    print(f"Pretext testing with tasks: {task1}, {task2}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    dataset = org.Organoids(file="utils/test.csv", task1=task1, task2=task2)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    logger = lg.Logger("pretext", task1, task2, evaluating=True)

    model_paths = get_pretrained_model(task1=task1, task2=task2)
    for fold, (model_path) in enumerate(model_paths):
        model = m.UNet(1, 1)

        # Load pretext trained model
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)

        ev.evaluate(task1, task2, fold, model, test_loader, device, logger)


if __name__ == '__main__':
    task1, task2 = get_args()

    test_pretext(task1=task1, task2=task2)
