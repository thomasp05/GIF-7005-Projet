import torch


def run_test(model, dataloader, tests, device):

    img_processed = 0

    for i, data in enumerate(dataloader):

        img, (labels, bounding_box) = data

        img = img.to(device)
        labels = labels.to(device)
        bounding_box = bounding_box.to(device)

        bounding_box_pred_ = model(img)

        if i == 0:
            # Pre-allocate results arrays
            real_bounding_box = torch.zeros(
                (len(dataloader.dataset), 1,
                 bounding_box.shape[-2], bounding_box.shape[-1])).to(device)
            pred_bounding_box = torch.zeros(real_bounding_box.shape).to(device)

            real_labels = torch.zeros((len(dataloader.dataset))).to(device)
            pred_labels = torch.zeros(real_labels.shape).to(device)

        real_bounding_box[img_processed:img.shape[0]] = bounding_box
        pred_bounding_box[img_processed:img.shape[0]] = bounding_box_pred_
        real_labels[img_processed:img.shape[0]] = labels
        pred_labels[img_processed:img.shape[0]
                    ] = bounding_box_pred_.flatten(-2, -1).max(-1).values.squeeze()

        for test in tests:

            test(real_bounding_box, pred_bounding_box, real_labels, pred_labels)


def IoU(bounding_box_1, bounding_box_2, *args):

    intersection = torch.logical_and(bounding_box_1, bounding_box_2).sum()
    union = torch.logical_or(bounding_box_1, bounding_box_2).sum()

    print("IoU : {}".format(100 * intersection.cpu() / union.cpu()))


def confusion_matrix(bb1, bb2, real_labels, pred_labels):

    mat = torch.zeros((2, 2))

    pred_labels[pred_labels > 0.5] = 1
    pred_labels[pred_labels < 0.5] = 0

    for i in range(2):
        for j in range(2):
            mat[i, j] = torch.logical_and(
                (pred_labels.flatten() == i), (real_labels.flatten() == j)).sum()

    print("Confusion matrix:")
    print(mat)
