import torch
import torch.nn as nn

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        conv_relu(in_channels, out_channels, 3, 1),
        conv_relu(in_channels, out_channels, 3, 1)
    )

def conv_relu(in_channels, out_channels, kernel, padding):
  return nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
    nn.ReLU(inplace=True)
  )