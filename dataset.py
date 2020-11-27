import copy
import os
import random

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms
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
            bounding_box[:, bb[1]:bb[1] + bb[3], bb[0]:bb[0] +
                         bb[2]] = torch.tensor(1.)

        if self.transforms:

            img, bounding_box = self.transforms(img, bounding_box)

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

class Crop():
    def __init__(self, max_shift):
        self.max_shift = max_shift
        
    def __call__(self, sample):
        origin, (target , mask) = sample
        tl_shift = np.random.randint(0, self.max_shift)
        br_shift = np.random.randint(0, self.max_shift)

        origin_w, origin_h = torchvision.transforms.functional.to_pil_image(origin).size
        crop_w = origin_w - tl_shift - br_shift
        crop_h = origin_h - tl_shift - br_shift
        
        origin = torchvision.transforms.functional.crop(origin, tl_shift, tl_shift,
                                                        crop_h, crop_w)
        mask = torchvision.transforms.functional.crop(mask, tl_shift, tl_shift,
                                                        crop_h, crop_w)
        return origin, (target, mask)

class Pad():
    def __init__(self, max_padding):
        self.max_padding = max_padding
        
    def __call__(self, sample):
        origin, (target , mask) = sample
        padding = np.random.randint(0, self.max_padding)

        origin = torchvision.transforms.functional.pad(origin, padding=padding, fill=0)
        mask = torchvision.transforms.functional.pad(mask, padding=padding, fill=0)
        return origin, (target, mask)

class Rotate():
    def __init__(self, max_angle):
        self.max_angle = max_angle
        
    def __call__(self, sample):
        origin, (target , mask) = sample
        angle = np.random.randint(-self.max_angle, self.max_angle)
        
        origin = torchvision.transforms.functional.rotate(origin, angle)
        mask = torchvision.transforms.functional.rotate(mask, angle)
        return origin, (target, mask)


class ImageTransform:
    def __init__(self ):
        self.image_transform = torchvision.transforms.Compose([
            Pad(200),
            Crop(100),
            Rotate(15)
        ])

    def __call__(self, sample):
        return self.image_transform(sample)

class DataAugmentation:
    def __init__(self, num_samples = 0):
        self.num_samples = num_samples
        self.transform = ImageTransform()

    def __call__(self, samples):
      new_dataset = []
      for s in samples:
        for n in range(self.num_samples):
          new_dataset.append(self.transform(s))
      samples = samples + new_dataset
      return samples