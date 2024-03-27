import torch
import argparse
import os
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

import src.downstream_model as dm
import src.downstream_train as dt
import src.organoids as org
import src.logger as lg


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

    return path + model_name


def train_downstream(task1="", task2=""):
    print(f"Downstream training with tasks: {task1}, {task2}")
    crossval_folds = 5
    epochs = 50
    batch_size = 16
    learning_rate = 0.001

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = dm.UNet(1, 1)
    loss_fn = dm.IoULoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    logger = lg.Logger("downstream", task1, task2)

    # Load pretrained task1 and task2 model
    pretrained = get_pretrained_model(task1=task1, task2=task2)
    model.load_state_dict(torch.load(pretrained))
    model.final_conv = torch.nn.Conv2d(64, 1, kernel_size=1)
    for i in range(len(model.down_conv)):
        model.down_conv[i].first.weight.requires_grad = False
        model.down_conv[i].second.weight.requires_grad = False
    model = model.to(device)

    dataset = org.Organoids(file="utils/downstream_train.csv")
    kf = KFold(n_splits=crossval_folds, shuffle=True, random_state=64)

    for fold, (train_index, val_index) in enumerate(kf.split(dataset)):
        print("Fold: ", fold)
        train_dataset = torch.utils.data.Subset(dataset, train_index)
        val_dataset = torch.utils.data.Subset(dataset, val_index)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        dt.downstream_train(model=model,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            optimizer=optimizer,
                            loss_fn=loss_fn,
                            fold=fold,
                            epochs=epochs,
                            device=device,
                            logger=logger)


def get_args():
    parser = argparse.ArgumentParser(description='Downstream Training')
    parser.add_argument('--task1', type=str, default="", help='Specify pretrained task to load for task 1')
    parser.add_argument('--task2', type=str, default="", help='Specify pretrained task to load for task 2')

    args = parser.parse_args()

    return args.task1, args.task2


if __name__ == '__main__':
    task1, task2 = get_args()

    train_downstream(task1=task1, task2=task2)
