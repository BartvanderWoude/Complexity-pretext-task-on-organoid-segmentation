import torch
from piqa import SSIM

import src.selfprediction_model as spm
import src.selfprediction_train as spt
import src.organoids as org


class SSIMLoss(SSIM):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return 1. - super().forward(x, y)


def pretext_train():
    epochs = 50
    batch_size = 16
    learning_rate = 0.001
    task1 = "D"
    task2 = ""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = spm.UNet(3, 3).to(device)
    loss_fn = SSIMLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    dataset = org.Organoids(file="utils/pretext_train.csv", task1=task1, task2=task2)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    spt.selfprediction_train(model, train_loader, optimizer, loss_fn, epochs, device)


if __name__ == '__main__':
    pretext_train()
