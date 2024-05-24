"""Model Trainer

author: Masahiro Hayashi

This script defines custom image transformations that simultaneously transform
both images and segmentation masks.

"""

import torchvision.transforms.functional as TF
# from torchvision.transforms import Compose

import torchvision.transforms.functional as TF
import numpy as np
import torch
from torch.utils.data import Dataset
import torch
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from albumentations import *

import cv2
import numpy as np
import random
import torch
import torchvision.transforms.functional as TF
from albumentations.augmentations.transforms import blur, GaussNoise, ISONoise, \
    RandomBrightnessContrast, RandomGamma, HueSaturationValue
from torchvision import transforms
from skimage.transform import resize
from scipy.ndimage import gaussian_filter, map_coordinates

class GaussianNoise:
    """Apply Gaussian noise to tensor."""

    def __init__(self, mean=0., std=1., p=0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, tensor):
        noise = 0
        if random.random() < self.p:
            noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'

class DoubleToTensor:
    """Apply horizontal flips to both image and segmentation mask."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask, weight=None):
        if weight is None:
            return TF.to_tensor(image), TF.to_tensor(mask)
        weight = weight.view(1, *weight.shape)
        return TF.to_tensor(image), TF.to_tensor(mask), weight

    def __repr__(self):
        return self.__class__.__name__ + '()'

class DoubleHorizontalFlip:
    """Apply horizontal flips to both image and segmentation mask."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask, weight=None):
        p = random.random()
        if p < self.p:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        if weight is None:
            return image, mask
        elif p > self.p:
            weight = TF.hflip(weight)
        return image, mask, weight

    def __repr__(self):
        return self.__class__.__name__ + f'(p={self.p})'

class DoubleVerticalFlip:
    """Apply vertical flips to both image and segmentation mask."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask, weight=None):
        p = random.random()
        if p < self.p:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        if weight is None:
            return image, mask
        elif p > self.p:
            weight = TF.hflip(weight)
        return image, mask, weight

    def __repr__(self):
        return self.__class__.__name__ + f'(p={self.p})'

class DoubleElasticTransform:
    """Based on implimentation on
    https://gist.github.com/erniejunior/601cdf56d2b424757de5"""

    def __init__(self, alpha=250, sigma=10, p=0.5, seed=None, randinit=True):
        if not seed:
            seed = random.randint(1, 100)
        self.random_state = np.random.RandomState(seed)
        self.alpha = alpha
        self.sigma = sigma
        self.p = p
        self.randinit = randinit


    def __call__(self, image, mask, weight=None):
        if random.random() < self.p:
            if self.randinit:
                seed = random.randint(1, 100)
                self.random_state = np.random.RandomState(seed)
                self.alpha = random.uniform(100, 300)
                self.sigma = random.uniform(10, 15)
                # print(self.alpha)
                # print(self.sigma)

            dim_image = image.shape
            dim_mask=mask.shape
            dx = self.alpha * gaussian_filter(
                (self.random_state.rand(*dim_image[1:]) * 2 - 1),
                self.sigma,
                mode="constant",
                cval=0
            )
            dy = self.alpha * gaussian_filter(
                (self.random_state.rand(*dim_image[1:]) * 2 - 1),
                self.sigma,
                mode="constant",
                cval=0
            )
            image = image.view(*dim_image[1:]).numpy()
            mask = mask.view(*dim_image[1:]).numpy()
            x, y = np.meshgrid(np.arange(dim_image[1]), np.arange(dim_image[2]))
            indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
            image = map_coordinates(image, indices, order=1)
            mask = map_coordinates(mask, indices, order=1)
            image, mask = image.reshape(dim_image), mask.reshape(dim_mask)
            image, mask = torch.Tensor(image), torch.Tensor(mask)
            if weight is None:
                return image, mask
            weight = weight.view(*dim_image[1:]).numpy()
            weight = map_coordinates(weight, indices, order=1)
            weight = weight.reshape(dim_image)
            weight = torch.Tensor(weight)

        return (image, mask) if weight is None else (image, mask, weight)


class DoubleCompose(transforms.Compose):

    def __call__(self, image, mask, weight=None):
        if weight is None:
            for t in self.transforms:
                image, mask = t(image, mask)
            return image, mask
        for t in self.transforms:
            image, mask, weight = t(image, mask, weight)
        return image, mask, weight






class DoubleBlur:
    """Apply blur to both image and segmentation mask."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            print("Image shape:", image.shape)
            print("Mask shape:", mask.shape)
            image = TF.gaussian_blur(image, kernel_size=3)
            mask = TF.gaussian_blur(mask, kernel_size=3)
        return image, mask

    def __repr__(self):
        return self.__class__.__name__ + f'(p={self.p})'

class DoubleGaussNoise:
    """Apply Gaussian noise to both image and segmentation mask."""

    def __init__(self, p=0.5, var_limit=(10.0, 50.0)):
        self.p = p
        self.var_limit = var_limit

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = image + torch.randn_like(image) * random.uniform(*self.var_limit)
            mask = mask + torch.randn_like(mask) * random.uniform(*self.var_limit)
        return image, mask

    def __repr__(self):
        return self.__class__.__name__ + f'(p={self.p}, var_limit={self.var_limit})'

class DoubleISONoise:
    """Apply ISO noise to both image and segmentation mask."""

    def __init__(self, p=0.5, intensity=(0.1, 0.5), color_shift=(0.01, 0.05)):
        self.p = p
        self.intensity = intensity
        self.color_shift = color_shift

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = image + torch.randn_like(image) * random.uniform(*self.intensity)
            mask = mask + torch.randn_like(mask) * random.uniform(*self.intensity)
        return image, mask

    def __repr__(self):
        return self.__class__.__name__ + f'(p={self.p}, intensity={self.intensity}, color_shift={self.color_shift})'

class DoubleRandomBrightnessContrast:
    """Apply random brightness and contrast to both image and segmentation mask."""

    def __init__(self, p=0.5, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)):
        self.p = p
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = TF.adjust_brightness(image, random.uniform(*self.brightness_limit))
            image = TF.adjust_contrast(image, random.uniform(*self.contrast_limit))
            mask = TF.adjust_brightness(mask, random.uniform(*self.brightness_limit))
            mask = TF.adjust_contrast(mask, random.uniform(*self.contrast_limit))
        return image, mask

    def __repr__(self):
        return self.__class__.__name__ + f'(p={self.p}, brightness_limit={self.brightness_limit}, ' \
                                         f'contrast_limit={self.contrast_limit})'

class DoubleRandomGamma:
    """Apply random gamma correction to both image and segmentation mask."""

    def __init__(self, p=0.5, gamma_limit=(80, 120), eps=1e-07):
        self.p = p
        self.gamma_limit = gamma_limit
        self.eps = eps

    def __call__(self, image, mask):
        if random.random() < self.p:
            gamma = random.uniform(*self.gamma_limit)
            image = TF.adjust_gamma(image, gamma)
            mask = TF.adjust_gamma(mask, gamma)
        return image, mask

    def __repr__(self):
        return self.__class__.__name__ + f'(p={self.p}, gamma_limit={self.gamma_limit}, eps={self.eps})'

class DoubleHueSaturationValue:
    """Apply random hue, saturation, and value shifts to both image and segmentation mask."""

    def __init__(self, p=0.5, hue_shift_limit=20, sat_shift_limit=10, val_shift_limit=10):
        self.p = p
        self.hue_shift_limit = hue_shift_limit
        self.sat_shift_limit = sat_shift_limit
        self.val_shift_limit = val_shift_limit

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = TF.adjust_hue(image, random.uniform(-self.hue_shift_limit, self.hue_shift_limit))
            image = TF.adjust_saturation(image, random.uniform(1, self.sat_shift_limit))
            image = TF.adjust_brightness(image, random.uniform(1, self.val_shift_limit))
            mask = TF.adjust_hue(mask, random.uniform(-self.hue_shift_limit, self.hue_shift_limit))
            mask = TF.adjust_saturation(mask, random.uniform(1, self.sat_shift_limit))
            mask = TF.adjust_brightness(mask, random.uniform(1, self.val_shift_limit))
        return image, mask

    def __repr__(self):
        return self.__class__.__name__ + f'(p={self.p}, hue_shift_limit={self.hue_shift_limit}, ' \
                                         f'sat_shift_limit={self.sat_shift_limit}, ' \
                                         f'val_shift_limit={self.val_shift_limit})'

class DoubleRotate:
    """Apply random rotation to both image and segmentation mask."""

    def __init__(self, p=1.0, angles=(0, 90, 180, 270)):
        self.p = p
        self.angles = angles

    def __call__(self, image, mask):
        if random.random() < self.p:
            angle = random.choice(self.angles)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)
        return image, mask

    def __repr__(self):
        return self.__class__.__name__ + f'(p={self.p}, angles={self.angles})'

