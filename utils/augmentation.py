"""
@augmentation
@reference: http://ceur-ws.org/Vol-2886/paper4.pdf
"""

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import albumentations as A
import matplotlib.pyplot as plt


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, img_gt):
        for t in self.transforms:
            img, img_gt = t(img, img_gt)
        return img, img_gt


class RandomRotate(object):
    def __init__(self, angle, p=0.5):
        self.p = p
        self.angle = angle

    def __call__(self, img, img_gt):
        if np.random.uniform(0, 1) < self.p:
            rotate_angle = np.random.uniform(self.angle[0], self.angle[1])
            img = img.rotate(rotate_angle, Image.BICUBIC)
            img_gt = img_gt.rotate(rotate_angle, Image.NEAREST)
        return img, img_gt


class RandomVerticalFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img, img_gt):
        if np.random.uniform(0, 1) < self.p:
            img = img.transpose(method=Image.FLIP_TOP_BOTTOM)
            img_gt = img_gt.transpose(method=Image.FLIP_TOP_BOTTOM)
        return img, img_gt


class RandomHorizontalFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img, img_gt):
        if np.random.uniform(0, 1) < self.p:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
            img_gt = img_gt.transpose(method=Image.FLIP_LEFT_RIGHT)
        return img, img_gt


class RandomGaussianNoise(object):
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img, img_gt):
        if np.random.uniform(0, 1) < self.p:
            img = A.GaussNoise(
                p=self.p, mean=0, var_limit=10)(image=np.array(img))['image']
            img = T.ToPILImage()(img)
        return img, img_gt


class ShiftScaleRotate(object):
    def __init__(self, p=0.3, shift_limit=0.0625, scale_limit=0.1, rotate_limit=10):
        self.p = p
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        self.rotate_limit = rotate_limit

    def __call__(self, img, img_gt):
        transform = A.ShiftScaleRotate(shift_limit=self.shift_limit,
                                       scale_limit=self.scale_limit, rotate_limit=self.rotate_limit, p=self.p)
        augmented = transform(image=np.array(img), mask=np.array(img_gt))
        img, img_gt = augmented['image'], augmented['mask']
        img = T.ToPILImage()(img)
        img_gt = T.ToPILImage()(img_gt)
        return img, img_gt


class MedianBlur(object):
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img, img_gt):
        img = A.MedianBlur(p=self.p)(
            image=np.array(img))['image']
        img = T.ToPILImage()(img)
        return img, img_gt


class Deformation(object):
    def __init__(self, p=0.3):
        self.p = p
    
    def __call__(self, img, img_gt):
        if np.random.uniform(0, 1) < self.p:
            transform = A.ElasticTransform(p=self.p, alpha=50, sigma=120 * 0.05, alpha_affine=120 * 0.03)
            augmented = transform(image=np.array(img), mask=np.array(img_gt))
            img, img_gt = augmented['image'], augmented['mask']
            img = T.ToPILImage()(img)
            img_gt = T.ToPILImage()(img_gt)
        return img, img_gt

class ColorJitter(object):
    def __init__(self, p=0.1, brightness=0.05, contrast=0.1, saturation=0.1, hue=0.05):
        self.p = p
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, img, img_gt):
        if np.random.uniform(0, 1) < self.p:
            transform = T.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue)
            img = transform(img)
        return img, img_gt

class MaskDropout(object):
    def __init__(self):
        pass

    def __call__(self, img, img_gt):
        pass
        return img, img_gt


class Resize(object):
    def __init__(self, input_size):
        self.input_size = input_size

    def __call__(self, img, img_gt):
        img = img.resize(
            (self.input_size[1], self.input_size[0]), Image.BICUBIC)
        img_gt = img_gt.resize(
            (self.input_size[1], self.input_size[0]), Image.NEAREST)
        return img, img_gt

class RandomCrop(object):
    def __init__(self, input_size):
        self.input_Size = input_size

    def __call__(self, img, img_gt):
        transform = A.RandomCrop(height=self.input_Size[0], width=self.input_Size[1])
        augmented = transform(image=np.array(img), mask=np.array(img_gt))
        img, img_gt = augmented['image'], augmented['mask']
        return T.ToPILImage()(img), T.ToPILImage()(img_gt)

class RandomScale(object):
    def __init__(self, scale_limit=0.2, p=1):
        self.p = p
        self.scale_limit = scale_limit

    def __call__(self, img, img_gt):
        if np.random.uniform(0, 1) < self.p:
            transform = A.RandomScale(scale_limit=self.scale_limit, p=self.p)
            augmented = transform(image=np.array(img), mask=np.array(img_gt))
            img, img_gt = augmented['image'], augmented['mask']
            img = T.ToPILImage()(img)
            img_gt = T.ToPILImage()(img_gt)
        return img, img_gt

class NormalizeTensor(object):
    def __call__(self, img, img_gt):
        img = T.ToTensor()(img)
        img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        img_gt = T.PILToTensor()(img_gt)
        return img, img_gt

    # img_gt = np.array(img_gt, dtype=np.int8)
    # masks = [(img_gt == v) for v in [1, 2]]
    # mask = np.stack(masks, axis=-1).astype('float')
    # img_gt = torch.from_numpy(mask)
    # img_gt = torch.permute(img_gt, (2, 0, 1)).contiguous()
