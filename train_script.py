import os
import sys
import time

os.chdir('/home/ludovicg/Documents/GIF-7005-Projet')

import torch
from dcm_dataset import *
from dcm_models import *

torch.manual_seed(111)
dataset = dcm_dataset('..')

train_set, test_set = train_test_split(dataset)

dataloader = torch.utils.data.DataLoader(
    train_set, batch_size=128, num_workers=2)


cnn = torch.nn.DataParallel(Vanilla_CNN(width=16).to("cuda:0"))
regressor = torch.nn.DataParallel(
    Simple_regressor(16 * 8 * 8).to("cuda:0"))

cnn.load_state_dict(torch.load("cnn.pt"))
regressor.load_state_dict(torch.load("regressor.pt"))


params = list(cnn.parameters()) + list(regressor.parameters())

loss_fn = torch.nn.BCEWithLogitsLoss()
optim = torch.optim.Adam(params, lr=1e-4)
n_epoch = 10

cnn.train()
regressor.train()

for epoch in range(n_epoch):
    timer = time.time()

    img_processed = 0

    for img, target in dataloader:

        optim.zero_grad()

        img, target = img.to("cuda:0"), target.to("cuda:0")

        pred1 = cnn(img)
        pred2 = regressor(pred1.flatten(1, -1)).squeeze()

        loss = loss_fn(pred2, target.to(torch.float))

        loss.backward()

        optim.step()

        img_processed += img.shape[0]

        # print("{:.2f} % done, {:.2f} s elapsed \r".format(
        #     100 * img_processed / len(dataloader.dataset), time.time() - timer))

    print("Epoch : {}".format(epoch + 1))
    print("Time elapsed : {}".format(time.time() - timer))

torch.save(cnn.state_dict(), "cnn.pt")
torch.save(regressor.state_dict(), "regressor.pt")

test_loader = torch.utils.data.DataLoader(
    train_set, batch_size=12, num_workers=2)


# Mesure le score

cnn.eval()
regressor.eval()

score = []

for img, target in test_loader:

    img, target = img.to("cuda:0"), target.to("cuda:0")

    pred1 = cnn(img)
    pred2 = regressor(pred1.flatten(1, -1)).squeeze()

    score_ = ((pred2 > 0.5) == (target == 1)).sum()

    score.append(score_)

score = torch.Tensor(score).sum() / len(test_loader.dataset)
print(score)
