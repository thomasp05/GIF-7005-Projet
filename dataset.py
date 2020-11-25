import copy
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from pydicom import dcmread


def train_test_split(dataset, test_size=0.2):

    test_set = copy.deepcopy(dataset)
    train_set = copy.deepcopy(dataset)

    if isinstance(test_size, float):
        test_size = int(len(dataset) * test_size)

    idx = torch.randperm(len(dataset))
    train_idx = idx[test_size:]
    test_idx = idx[:test_size]

    test_set.id = test_set.id[test_idx]
    test_set.update()

    train_set.id = train_set.id[train_idx]
    train_set.update()

    return train_set, test_set


class dcm_dataset(torch.utils.data.Dataset):
    def __init__(self, directory, transforms=None):
        """
        directory needs to contain the 'stage_2_train_labels.csv' file and the
        'stage_2_train_images' subdirectory
        """

        self.directory = directory

        img_directory = directory + '/' + 'stage_2_train_images'
        self.img_files = self.explore_directory(img_directory)

        self.data_csv = np.genfromtxt(
            directory + '/' + 'stage_2_train_labels.csv',
            delimiter=',', skip_header=1, dtype='str')

        self.id = np.unique(self.data_csv[:, 0])
        self.targets = []
        self.img_idx = []

        self.update()

        self.transforms = transforms

    def __len__(self):

        return len(self.id)

    def __getitem__(self, index):

        patient_id = self.id[index]
        targets = self.targets[index]
        img_file = self.img_files[self.img_idx[index]]

        img = torch.tensor(dcmread(img_file).pixel_array).unsqueeze(
            0).to(torch.float)
        target = torch.tensor(targets[0]).to(torch.float)

        bounding_box = torch.zeros(img.shape).to(torch.float)

        for bb in targets[1]:
        return img, (target, bounding_box)

    def display(self):

        idx = np.random.randint(0, len(self), 8)

        for i in range(idx.shape[0]):

            plt.subplot(int('24{}'.format(i + 1)))

            img, (target, _) = self[idx[i]]
            _, bounding_box = self.targets[idx[i]]

            plt.imshow(img.squeeze())
            ax = plt.gca()

            if int(target) == 1:
                for bounding_box_ in bounding_box:
                    x_min, y_min, width, height = bounding_box_

                    rect = patches.Rectangle((float(x_min), float(y_min)),
                                             float(width), float(height),
                                             linewidth=2, edgecolor='r',
                                             facecolor='none')
                    ax.add_patch(rect)
            plt.title(target)
            plt.axis('off')

        plt.tight_layout()

        plt.show()

    def explore_directory(self, directory, ext='.dcm'):

        if directory[-1] in ['/']:
            directory = directory[:-1]

        content = os.listdir(directory)

        files_found = []

        for file in content:
            if os.path.isdir(directory + '/' + file):
                # Found a subdirectory
                sub_content = self.explore_directory(directory + '/' + file)

                for sub_file in sub_content:

                    files_found.append(sub_file)

            elif os.path.isfile(directory + '/' + file):
                if ext in file:

                    files_found.append(directory + '/' + file)

        return files_found

    def update(self):

        self.targets = []
        self.img_idx = []

        for id in self.id:
            # Récupérer le "dossier" de ce patient
            info = self.data_csv[(self.data_csv[:, 0] == id)]
            target = info[:, -1].astype(np.int).min()
            if target == 0:
                bounding_box = []
            else:
                bounding_box = info[:, 1:-1].astype(np.float).astype(np.int)
            self.targets.append((target, bounding_box))

            # Trouver l'indice de l'image associée
            for idx in range(len(self.img_files)):
                if id in self.img_files[idx]:
                    self.img_idx.append(idx)
                    continue

    def subset(self, fraction=0.25):
        idx_to_keep = torch.randint(
            0, len(self.id), (int(fraction * len(self.id)), 1)).squeeze()

        self.id = self.id[idx_to_keep]
        self.update()

        
class Downsample:
  def __init__(self):
    self.pool = torch.nn.AvgPool2d(2)
    
  def __call__(self, x, target):
    x = self.pool(x.unsqueeze(0)).squeeze(0)
    target = self.pool(target)
    return x, target
