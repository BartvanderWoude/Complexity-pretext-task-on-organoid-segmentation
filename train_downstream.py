import src.model as m
import src.train as t
import src.organoids as org
import src.logger as lg

import torch
import argparse
import os
import pandas as pd
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader


def get_args():
    parser = argparse.ArgumentParser(description='Downstream Training')
    parser.add_argument('--task1', type=str, default="", help='Specify pretrained task to load for task 1')
    parser.add_argument('--task2', type=str, default="", help='Specify pretrained task to load for task 2')
    parser.add_argument('--dummy', type=bool, default=False, help='Use dummy data')

    args = parser.parse_args()

    return args.task1, args.task2, args.dummy


def get_pretrained_model(task1="", task2=""):
    model_path = f"output/models/pretext/{task1}_{task2}/"
    evaluation_path = "output/evaluation/pretext_evaluation.csv"

    if not os.path.exists(evaluation_path):
        raise ValueError("No pretext evaluations found.")

    if not os.path.exists(model_path):
        raise ValueError("No downstream models found.")

    df = pd.read_csv(evaluation_path)
    df["task2"] = df["task2"].fillna("")

    df = df[(df["task1"] == task1) & (df["task2"] == task2)]
    df = df.sort_values(by="psnr", ascending=False)
    highest_fold = df.iloc[0]["fold"]

    model_name = f"f{highest_fold}_pretext_{task1}_{task2}.pth"

    print(f"Loading model: {model_name}")

    return model_path + model_name


def initialize_model():
    model = m.UNet(1, 1)

    pretrained = get_pretrained_model(task1=task1, task2=task2)
    model.load_state_dict(torch.load(pretrained), strict=False)

    model.final_conv = torch.nn.Conv2d(64, 1, kernel_size=1)
    for i in range(len(model.down_conv)):
        model.down_conv[i].first.weight.requires_grad = False
        model.down_conv[i].second.weight.requires_grad = False

    return model


def train_downstream(task1, task2, use_dummy):
    print(f"Downstream training with tasks: {task1}, {task2}")
    crossval_folds = 5
    epochs = 50
    batch_size = 16
    learning_rate = 0.001

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = initialize_model().to(device)
    loss_fn = m.IoULoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    logger = lg.Logger("downstream", task1, task2)

    if use_dummy:
        csv_path = "utils/dummy.csv"
        data_path = "dummy_data/"
    else:
        csv_path = "utils/downstream_train.csv"
        data_path = "organoid_data/"

    dataset = org.Organoids(csv_path=csv_path, data_path=data_path)
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
    task1, task2, use_dummy = get_args()

    train_downstream(task1=task1, task2=task2, use_dummy=use_dummy)
