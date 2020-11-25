import torch


def run_test(model, dataloader, tests, device):

    img_processed = 0

    for i, data in enumerate(dataloader):

        img, (labels, bounding_box) = data

        img = img.to(device)
        labels = labels.to(device)
        bounding_box = bounding_box.to(device)

        bounding_box_pred = model(img)
        bounding_box_pred[bounding_box_pred > 0.5] = 1
        bounding_box_pred[bounding_box_pred < 0.5] = 0

        pred_labels = bounding_box_pred.flatten(
            -2, -1).max(-1).values.squeeze()

        for test in tests:

            test(bounding_box, bounding_box_pred, labels, pred_labels)

    for test in tests:

        test.compute()


class IoU:

    def __init__(self):

        self.iou = []
        self.weights = []
        self.result = None

    def __call__(self, bb1, bb2, *args):

        intersection = torch.logical_and(
            bb1.detach(), bb2.detach()).sum()
        union = torch.logical_or(bb1.detach(),
                                 bb2.detach()).sum()

        self.iou.append(100 * intersection.cpu() / union.cpu())
        self.weights.append(bb1.shape[0])

    def __str__(self):

        print(self.result)

    def compute(self):

        self.result = (torch.tensor(self.iou) * torch.tensor(self.weights)
                       ).sum() / torch.tensor(self.weights).sum()

        print("IoU : {} %".format(self.result))


class Confusion_matrix:

    def __init__(self):

        self.mat = None
        self.result = None

    def __call__(self, bb1, bb2, real_labels, pred_labels):

        mat_ = torch.zeros((2, 2))

        pred_labels[pred_labels > 0.5] = 1
        pred_labels[pred_labels < 0.5] = 0

        for i in range(2):
            for j in range(2):
                mat_[i, j] = torch.logical_and(
                    (pred_labels.flatten() == i), (real_labels.flatten() == j)).sum()

        if self.mat == None:
            self.mat = mat_.unsqueeze(0).cpu()

        else:
            self.mat = torch.cat([self.mat, mat_.unsqueeze(0).cpu()], 0)

    def __str__(self):

        print(self.result)

    def compute(self):

        self.result = self.mat.sum(0)

        print(self.result)
