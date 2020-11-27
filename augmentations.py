import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms


class Downsample:

    def __init__(self, factor=2):
        self.pool = nn.AvgPool2d(factor)

    def __call__(self, sample):

        origin, (target, mask) = sample

        origin = self.pool(origin)
        mask = self.pool(mask)
        return origin, (target, mask)


class Crop:

    def __init__(self, max_shift):
        self.max_shift = max_shift

    def __call__(self, sample):
        origin, (target, mask) = sample
        tl_shift = np.random.randint(0, self.max_shift)
        br_shift = np.random.randint(0, self.max_shift)

        origin_w, origin_h = torchvision.transforms.functional.to_pil_image(
            origin).size
        crop_w = origin_w - tl_shift - br_shift
        crop_h = origin_h - tl_shift - br_shift

        origin = torchvision.transforms.functional.crop(origin, tl_shift, tl_shift,
                                                        crop_h, crop_w)
        mask = torchvision.transforms.functional.crop(mask, tl_shift, tl_shift,
                                                      crop_h, crop_w)
        return origin, (target, mask)


class Pad:

    def __init__(self, max_padding):
        self.max_padding = max_padding

    def __call__(self, sample):
        origin, (target, mask) = sample
        padding = np.random.randint(0, self.max_padding)

        origin = torchvision.transforms.functional.pad(
            origin, padding=padding, fill=0)
        mask = torchvision.transforms.functional.pad(
            mask, padding=padding, fill=0)
        return origin, (target, mask)


class Rotate:

    def __init__(self, max_angle):
        self.max_angle = max_angle

    def __call__(self, sample):
        origin, (target, mask) = sample
        angle = np.random.randint(-self.max_angle, self.max_angle)

        origin = torchvision.transforms.functional.rotate(origin, angle)
        mask = torchvision.transforms.functional.rotate(mask, angle)
        return origin, (target, mask)


class ImageTransform:

    def __init__(self):
        self.image_transform = torchvision.transforms.Compose([
            Pad(100),
            Crop(100),
            Rotate(15)
        ])

    def __call__(self, sample):
        return self.image_transform(sample)


class DataAugmentation:

    def __init__(self, num_samples=0):
        self.num_samples = num_samples
        self.transform = ImageTransform()

    def __call__(self, samples):
        new_dataset = []
        for s in samples:
            for n in range(self.num_samples):
                new_dataset.append(self.transform(s))
        samples = samples + new_dataset
        return samples
