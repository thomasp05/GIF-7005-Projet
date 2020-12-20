import torch
import torch.nn as nn
import torchvision.models
from models_parts import *


class CNN_block(nn.Module):

    def __init__(self, in_channels, out_channels):

        super(CNN_block, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, [3, 3], 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.MaxPool2d([2, 2], 2)
        )

    def forward(self, x):

        return self.layer(x)


class Vanilla_CNN(nn.Module):

    def __init__(self, width):

        super(Vanilla_CNN, self).__init__()

        self.block1 = CNN_block(1, int(width * 0.5))
        self.block2 = CNN_block(int(width * 0.5), width)
        self.block3 = CNN_block(width, width)
        self.block4 = CNN_block(width, width)
        self.block5 = CNN_block(width, width)
        self.block6 = CNN_block(width, width)
        self.block7 = CNN_block(width, width)

    def forward(self, x):

        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.block7(out)

        return out


class Simple_regressor(nn.Module):

    def __init__(self, in_channels, out_channels=1):

        super(Simple_regressor, self).__init__()

        self.layer = nn.Linear(in_channels, out_channels)

    def forward(self, x):

        return self.layer(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class ResNetUNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()

        self.base_model = torchvision.models.resnet18(pretrained=True)

        avg_weights = torch.mean(self.base_model.conv1.weight, 1, True)
        self.base_model.conv1 = nn.Conv2d(
            1, 64, 7, stride=2, padding=3, bias=False)
        self.base_model.conv1.weight = nn.Parameter(avg_weights)

        self.base_layers = list(self.base_model.children())

        # size=(N, 64, x.H/2, x.W/2)
        self.layer0 = nn.Sequential(*self.base_layers[:3])
        self.layer0_1x1 = conv_relu(64, 64, 1, 0)
        # size=(N, 64, x.H/4, x.W/4)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer1_1x1 = conv_relu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = conv_relu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = conv_relu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = conv_relu(512, 512, 1, 0)

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = conv_relu(256 + 512, 512, 3, 1)
        self.conv_up2 = conv_relu(128 + 512, 256, 3, 1)
        self.conv_up1 = conv_relu(64 + 256, 256, 3, 1)
        self.conv_up0 = conv_relu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = conv_relu(1, 64, 3, 1)
        self.conv_original_size1 = conv_relu(64, 64, 3, 1)
        self.conv_original_size2 = conv_relu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):

        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out


class Inception_v3(nn.Module):

    def __init__(self, pretrained=True):

        super().__init__()
        self._return_grad = False

        # resample to 299 x 299 spatial sizes
        self.upsample = nn.Upsample(
            size=(299, 299), mode='bilinear', align_corners=True)

        self.upsample_grad = nn.Upsample(
            size=(512, 512), mode='bilinear', align_corners=True)

        self.relu = nn.ReLU()

        self.base_model = torchvision.models.inception_v3(
            pretrained=pretrained)

        # freeze everything but the last layer
        for param in self.base_model.parameters():
            param.requires_grad = not pretrained

        self.base_model.fc = nn.Linear(2048, 1, bias=True)
        self.base_model.AuxLogits.fc = nn.Linear(768, 1, bias=True)

    def forward(self, x):

        out = self.upsample(x)

        # To simulate 3 channels
        if not self._return_grad:
            out = torch.cat([out, out, out], dim=1)
            out = self.base_model(out)
        else:
            out = self.to_grad(x)

        return out

    def to_fmap(self, x):

        x = self.upsample(x)
        x = torch.cat([x, x, x], dim=1)

        x = self.base_model.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.base_model.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.base_model.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.base_model.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.base_model.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.base_model.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.base_model.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.base_model.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.base_model.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.base_model.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.base_model.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.base_model.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.base_model.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.base_model.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.base_model.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = self.base_model.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.base_model.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.base_model.Mixed_7c(x)
        # N x 2048 x 8 x 8

        return x

    def to_pred(self, x):

        x = self.base_model.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.base_model.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        x = self.base_model.fc(x)
        # N x 1000 (num_classes)
        return x

    def to_grad(self, x):

        fmap = self.to_fmap(x).detach()
        fmap.requires_grad = True

        pred = self.to_pred(fmap)
        pred.backward()

        grad = self.upsample_grad(
            self.relu(fmap.grad * fmap.detach()
                      ).sum(dim=1, keepdims=True))

        return grad

    def return_grad(self, bool_=True):

        self._return_grad = True


class FCN(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        
        self.base_model = torchvision.models.resnet18(pretrained=True)

        avg_weights = torch.mean(self.base_model.conv1.weight, 1, True)
        self.base_model.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        self.base_model.conv1.weight = nn.Parameter(avg_weights)
        
        self.base_layers = list(self.base_model.children())
        self.layer1 = nn.Sequential(*self.base_layers[:5]) # size=(N, 64, x.H/2, x.W/2)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/4, x.W/4)
        self.upsample2 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/8, x.W/8)
        self.upsample3 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/16, x.W/16)
        self.upsample4 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        
        self.conv1k = nn.Conv2d(64 + 128 + 256 + 512, n_class, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.layer1(x)
        up1 = self.upsample1(x)
        x = self.layer2(x)
        up2 = self.upsample2(x)
        x = self.layer3(x)
        up3 = self.upsample3(x)
        x = self.layer4(x)
        up4 = self.upsample4(x)
        
        merge = torch.cat([up1, up2, up3, up4], dim=1)
        merge = self.conv1k(merge)
        out = self.sigmoid(merge)
        
        return out        


class DenseNet121(nn.Module):

    def __init__(self, pretrained=True):

        super().__init__()

        # resample to 224 x 224 spatial sizes
        self.upsample = nn.Upsample(
            size=(224, 224), mode='bilinear', align_corners=True)

        self.base_model = torch.hub.load(
            'pytorch/vision:v0.6.0', 'densenet121', pretrained=True)

        # reshape the ouput layer
        self.base_model.classifier = nn.Linear(1024, 1)
        self.base_model.classifier.requires_grad = True

        # freeze everything but the last layer
        for name, param in self.named_parameters():
            if "classifier" in name:
                param.requires_grad = True
            else:
                param.requires_grad = not pretrained

    def forward(self, x):

        out = self.upsample(x)

        # To simulate 3 channels
        out = torch.cat([out, out, out], dim=1)
        out = self.base_model(out)

        return out
