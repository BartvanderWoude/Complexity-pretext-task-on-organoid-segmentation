import src.model as m
import src.train as t
import src.organoids as org
import src.logger as lg

import torch
import argparse
from torchvision import models
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader


def get_args():
    parser = argparse.ArgumentParser(description='Pretext Training')
    parser.add_argument('--task1', type=str, default="", help='Specify distortion type for task 1')
    parser.add_argument('--task2', type=str, default="", help='Specify distortion type for task 2')

    args = parser.parse_args()

    return args.task1, args.task2


def initialize_model():
    model = m.UNet(1, 1)

    resnet = models.resnet18(weights="IMAGENET1K_V1")
    top_layers = list(resnet.children())

    for i in range(1, 4):
        basicblocks = list(top_layers[i+4].children())
        component = list(basicblocks[0].children())
        model.down_conv[i].first.weight = component[0].weight
        model.down_conv[i].second.weight = component[3].weight

    return model


def train_pretext(task1="", task2=""):
    print(f"Pretext training with tasks: {task1}, {task2}")
    crossval_folds = 5
    epochs = 50
    batch_size = 16
    learning_rate = 0.001

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = initialize_model().to(device)
    loss_fn = m.SSIMLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    logger = lg.Logger("pretext", task1, task2)

    dataset = org.Organoids(file="utils/pretext_train.csv", task1=task1, task2=task2)
    kf = KFold(n_splits=crossval_folds, shuffle=True, random_state=64)

    for fold, (train_index, val_index) in enumerate(kf.split(dataset)):
        print("Fold: ", fold)

        train_dataset = torch.utils.data.Subset(dataset, train_index)
        val_dataset = torch.utils.data.Subset(dataset, val_index)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        t.train(model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                fold=fold,
                epochs=epochs,
                device=device,
                logger=logger)


if __name__ == '__main__':
    task1, task2 = get_args()

    train_pretext(task1=task1, task2=task2)
