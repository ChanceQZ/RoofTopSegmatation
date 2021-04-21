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

IMAGE_SIZE = 256
CROP_SIZE = 256

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
    def __init__(self, image_paths, mask_paths, transform=trfm, mode="train"):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.mode = mode

        self.len = len(image_paths)

        self.valid_trfm = T.Compose([
            T.ToPILImage(),
            T.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5],
                        [0.5, 0.5, 0.5])
        ])

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
        if self.mode.lower() == "train":
            mask = cv2.imread(self.mask_paths[index], cv2.IMREAD_GRAYSCALE)
            augments = self.transform(image=img, mask=mask)
            return self.as_tensor(augments["image"]), augments["mask"][None]  # "None" can add 1st dimension
        elif self.mode.lower() == "valid":
            mask = cv2.imread(self.mask_paths[index], cv2.IMREAD_GRAYSCALE)
            return self.valid_trfm(img), mask[None]
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

    train_ds = RoofTopDataset(image_paths, mask_paths)
    valid_ds = RoofTopDataset(image_paths, mask_paths, mode="valid")

    return train_ds, valid_ds


def get_test_data():
    pass

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

    loader = D.DataLoader(
        rf_ds, batch_size=16, shuffle=True, num_workers=0)
    image, mask = next(iter(loader))
    print(image.shape)
    print(mask.shape)
    """

    plt.figure(figsize=(16, 8))
    plt.subplot(121)
    plt.imshow(np.where(mask==1, 255, 0), cmap='gray')
    plt.subplot(122)
    plt.imshow(img[0])
    plt.show()
"""
