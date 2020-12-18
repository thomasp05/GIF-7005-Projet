import torch
import torch.nn as nn


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, residual_connection=True):

        super().__init__()

        self.residual_connection = residual_connection

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, [
                               3, 3], stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, [
                               3, 3], stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.residual_connection:

            out += identity

        return out


class Resnet_cam(nn.Module):

    def __init__(self):

        super().__init__()

        self.layers = nn.Sequential(
            BasicBlock(1, 64, False),
            BasicBlock(64, 64),
            BasicBlock(64, 128, False),
            BasicBlock(128, 128),
            nn.MaxPool2d(2),
            BasicBlock(128, 256, False),
            BasicBlock(256, 256),
            nn.MaxPool2d(2),
            BasicBlock(256, 512, False),
            BasicBlock(512, 512),
            nn.MaxPool2d(2),
            BasicBlock(512, 512),
            nn.MaxPool2d(2),
            BasicBlock(512, 512))

        self.avg_pool = nn.AvgPool2d(32)

        self.flatten = nn.Flatten(1, -1)

        self.fc = nn.Linear(512, 1)

        self.upsample = nn.Upsample(
            scale_factor=16, mode='bilinear', align_corners=True)

        self.relu = nn.ReLU()

        self._return_fmap = False

    def forward(self, x):

        if not self._return_fmap:

            out = self.layers(x)
            out = self.avg_pool(out)
            out = self.flatten(out)
            out = self.fc(out)

        else:
            out = self.grad_analysis(x)

        return out

    def get_feature_map(self, x):

        out = self.layers(x)

        return out

    def feature_map_to_prediction(self, x):

        out = self.avg_pool(x)
        out = self.flatten(out)
        out = self.fc(out)

        return out

    def grad_analysis(self, x):

        fmap = self.get_feature_map(x).detach()
        fmap.requires_grad = True

        pred = self.feature_map_to_prediction(fmap)
        pred.backward()

        grad = self.upsample(self.relu(fmap.grad * fmap.detach()
                                       ).sum(dim=1, keepdims=True))

        return grad

    def return_fmap(self, value=True):

        self._return_fmap = value
