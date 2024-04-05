import src.model as m
import src.organoids as org
import src.logger as lg
import src.evaluation as ev

import torch
import torch.nn as nn
import argparse
import os
from torch.utils.data import DataLoader


def get_args():
    parser = argparse.ArgumentParser(description='Pretext Testing')
    parser.add_argument('--task1', type=str, default="", help='Specify distortion type for task 1')
    parser.add_argument('--task2', type=str, default="", help='Specify distortion type for task 2')
    parser.add_argument('--dummy', type=bool, default=False, help='Use dummy data')

    args = parser.parse_args()

    return args.task1, args.task2, args.dummy


def get_pretrained_models(task1="", task2=""):
    model_path = f"output/models/pretext/{task1}_{task2}/"

    if not os.path.exists(model_path):
        raise ValueError("No pretext models found.")

    models = os.listdir(model_path)
    models = sorted(models)
    model_paths = [model_path + model for model in models]

    print(f"Loading models: {model_paths}")

    return model_paths


def initialize_model(task1=""):
    if task1 == "j":
        model = m.UNetInnate(1, 1, 9)
    elif task1 == "p":
        model = m.UNetInnate(1, 1, 4)
        model.fc3 = nn.Linear(256, 4)
        model.reshape = nn.Unflatten(1, (4,))
    else:
        model = m.UNet(1, 1)

    return model


def test_pretext(task1, task2, use_dummy):
    print(f"Pretext testing with tasks: {task1}, {task2}")
    batch_size = 16

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if use_dummy:
        csv_path = "utils/dummy.csv"
        data_path = "dummy_data/"
    else:
        csv_path = "utils/test.csv"
        data_path = "organoid_data/"

    dataset = org.Organoids(csv_path=csv_path, data_path=data_path, task1=task1, task2=task2)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    logger = lg.Logger("pretext", task1, task2, evaluating=True)

    model_paths = get_pretrained_models(task1=task1, task2=task2)
    for fold, (model_path) in enumerate(model_paths):
        model = initialize_model(task1=task1)
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)

        ev.evaluate(task1=task1,
                    task2=task2,
                    fold=fold,
                    model=model,
                    test_loader=test_loader,
                    device=device,
                    logger=logger)


if __name__ == '__main__':
    task1, task2, use_dummy = get_args()

    test_pretext(task1=task1, task2=task2, use_dummy=use_dummy)
