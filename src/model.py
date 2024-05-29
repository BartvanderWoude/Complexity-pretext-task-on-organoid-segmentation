# Source U-Net: Lab ML - https://nn.labml.ai/unet/index.html
# https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/unet/__init__.py

# Source IoU loss: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch

import src.unet_blocks as ub

import torch
from torch import nn
from piqa import SSIM


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth)/(union + smooth)

        return 1 - IoU


class SSIMLoss(SSIM):
    def __init__(self):
        super().__init__(n_channels=1)

    def forward(self, x, y):
        return 1. - super().forward(x, y)


class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.down_conv = nn.ModuleList([ub.DoubleConvolution(i, o) for i, o in
                                        [(in_channels, 64), (64, 128), (128, 256), (256, 512)]])
        self.down_sample = nn.ModuleList([ub.DownSample() for _ in range(4)])

        self.middle_conv = ub.DoubleConvolution(512, 1024)

        self.up_sample = nn.ModuleList([ub.UpSample(i, o) for i, o in
                                        [(1024, 512), (512, 256), (256, 128), (128, 64)]])
        self.up_conv = nn.ModuleList([ub.DoubleConvolution(i, o) for i, o in
                                      [(1024, 512), (512, 256), (256, 128), (128, 64)]])

        self.concat = nn.ModuleList([ub.CropAndConcat() for _ in range(4)])

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        pass_through = []

        for i in range(len(self.down_conv)):
            x = self.down_conv[i](x)

            pass_through.append(x)

            x = self.down_sample[i](x)

        x = self.middle_conv(x)

        for i in range(len(self.up_conv)):
            x = self.up_sample[i](x)

            x = self.concat[i](x, pass_through.pop())

            x = self.up_conv[i](x)

        x = self.final_conv(x)
        x = self.sigmoid(x)

        return x


class UNetInnate(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_classes: int):
        super().__init__()

        self.down_conv = nn.ModuleList([ub.DoubleConvolution(i, o) for i, o in
                                        [(in_channels, 64), (64, 128), (128, 256), (256, 512)]])
        self.down_sample = nn.ModuleList([ub.DownSample() for _ in range(4)])

        self.middle_conv = ub.DoubleConvolution(512, 1024)

        self.up_sample = nn.ModuleList([ub.UpSample(i, o) for i, o in
                                        [(1024, 512), (512, 256), (256, 128), (128, 64)]])
        self.up_conv = nn.ModuleList([ub.DoubleConvolution(i, o) for i, o in
                                      [(1024, 512), (512, 256), (256, 128), (128, 64)]])

        self.concat = nn.ModuleList([ub.CropAndConcat() for _ in range(4)])

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.fc1 = nn.Linear(102400, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_classes*n_classes)
        self.reshape = nn.Unflatten(1, (n_classes, n_classes, ))

    def forward(self, x: torch.Tensor):
        pass_through = []

        for i in range(len(self.down_conv)):
            x = self.down_conv[i](x)

            pass_through.append(x)

            x = self.down_sample[i](x)

        x = self.middle_conv(x)

        for i in range(len(self.up_conv)):
            x = self.up_sample[i](x)

            x = self.concat[i](x, pass_through.pop())

            x = self.up_conv[i](x)

        x = self.final_conv(x)

        x = torch.flatten(x, 1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.reshape(x)

        return x
