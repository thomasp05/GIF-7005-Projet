import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
from pydicom import dcmread

train_labels_csv = '../stage_2_train_labels.csv'
train_images_folder = '../stage_2_train_images'


def train_test_split(dataset, test_size=0.2, shuffle=True):

    test_set = copy.deepcopy(dataset)
    train_set = copy.deepcopy(dataset)

    if isinstance(test_size, float):
        test_size = int(len(dataset) * test_size)

    data_csv = dataset.data_csv

    if shuffle:
        data_csv = data_csv[torch.randperm(data_csv.shape[0])]

    test_set.data_csv = data_csv[:test_size, :]
    train_set.data_csv = data_csv[test_size:, :]

    return train_set, test_set


class dcm_dataset(torch.utils.data.Dataset):
    def __init__(self, directory, transforms=None):
        """
        directory needs to contain the 'stage_2_train_labels.csv' file and the
        'stage_2_train_images' subdirectory
        """

        self.directory = directory

        self.img_directory = directory + '/' + 'stage_2_train_images'

        self.data_csv = np.genfromtxt(
            directory + '/' + 'stage_2_train_labels.csv',
            delimiter=',', skip_header=1, dtype='str')

        self.transforms = transforms

    def __len__(self):

        return self.data_csv.shape[0]

    def __getitem__(self, index):

        img = torch.Tensor(dcmread(self.img_directory + '/' +
                                   self.data_csv[index, 0] + '.dcm').pixel_array).unsqueeze(0)

        target = self.data_csv[index, -1].astype(np.int)

        if self.transforms:

            img, target = self.transforms(img, target)

        return img, target


if __name__ == "__main__":

    # Display some examples

    dset = dcm_dataset('..')

    idx = np.random.randint(0, len(dset), 8)

    # plt.figure(figsize=[10, 2])

    for i in range(idx.shape[0]):

        plt.subplot(int('24{}'.format(i + 1)))

        img, target = dset[idx[i]]

        plt.imshow(img)
        plt.title(target)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
