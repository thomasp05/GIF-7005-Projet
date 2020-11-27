import torch
import torch.nn as nn


class Grad_analyser:

    def __init__(self, model):
        """
        model : callable
         Takes takes as entry a batch of images of shape (N, C, H, W) and returns
         a score for the whole image, of shape (N, 1).
        """

        self.model = model

        # Make sure weights are frozen
        for param in self.model.parameters():
            param.requires_grad = False

        self.criterion = nn.BCELoss()

    def __call__(self, x):
        """
        x is an image batch of shape (N, C, H, W)
        """

        x.requires_grad = True

        pred = self.model(x)

        # This part is just a placeholder
        loss = self.criterion(pred, torch.ones(pred.shape).to(x.device))

        loss.backward()

        img = x.grad

        return img
