# -*- coding: utf-8 -*-

"""
@File: roottop_dataset.py
@Author: Chance (Qian Zhen)
@Description: 
@Date: 4/20/2021
"""

import os
import glob
import cv2
import albumentations as A
from torchvision import transforms as T
import torch.utils.data as D

IMAGE_SIZE = 512
CROP_SIZE = 512

trfm = A.Compose([
    A.RandomCrop(CROP_SIZE, CROP_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomGamma(p=0.5),
    A.GaussNoise(p=0.5),
    A.ISONoise(p=0.5),
    A.Blur(blur_limit=3, p=0.5)
])


class RoofTopDataset(D.Dataset):
    def __init__(self, image_paths, mask_paths, transform=trfm, test_mode=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.test_mode = test_mode

        self.len = len(image_paths)
        self.as_tensor = T.Compose([
            T.ToPILImage(),
            T.Resize(IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5],
                        [0.5, 0.5, 0.5]),
        ])

    # get data operation
    def __getitem__(self, index):
        img = cv2.imread(self.image_paths[index])
        if not self.test_mode:
            mask = cv2.imread(self.mask_paths[index])
            augments = self.transform(image=img, mask=mask)
            return self.as_tensor(augments['image']), augments['masks']
        else:
            return self.as_tensor(img), ''

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len

def get_train_valid_data(image_folder, mask_folder, split_rate=10):
    image_paths = glob.glob(os.path.join(image_folder, "*.png"))
    mask_paths = glob.glob(os.path.join(mask_folder, "*.png"))

    rf_ds = RoofTopDataset(image_paths, mask_paths)

    train_idx, valid_idx = [], []
    for i in range(len(rf_ds)):
        if i % split_rate == 0:
            valid_idx.append(i)
        else:
            train_idx.append(i)

    train_ds = D.Subset(rf_ds, train_idx)
    valid_ds = D.Subset(rf_ds, valid_idx)
    return train_ds, valid_ds

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    image_paths = glob.glob("./data/train/images/*.png")
    mask_paths = glob.glob("data/train/masks/*.png")

    rf_ds = RoofTopDataset(image_paths, mask_paths, trfm)
    print(len(rf_ds))
    img, mask = rf_ds[0]
    print(img.shape)
    print(mask.shape)

    plt.figure(figsize=(16, 8))
    plt.subplot(121)
    plt.imshow(np.where(mask==1, 255, 0), cmap='gray')
    plt.subplot(122)
    plt.imshow(img[0])
    plt.show()
