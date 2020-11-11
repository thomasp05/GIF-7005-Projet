import torch


class CNN_block(torch.nn.Module):

    def __init__(self, in_channels, out_channels):

        super(CNN_block, self).__init__()

        self.layer = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, [3, 3], 1, 1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU()
        )

    def forward(self, x):

        return self.layer(x)


class Vanilla_CNN(torch.nn.Module):

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


class Simple_regressor(torch.nn.Module):

    def __init__(self, in_channels, out_channels=1):

        super(Simple_regressor, self).__init__()

        self.layer = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x):

        return self.layer(x)
