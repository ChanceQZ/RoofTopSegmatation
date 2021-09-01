# -*- coding: utf-8 -*-

"""
@File: rooftop_dataset.py
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
import concurrent.futures
from utils import load_img_mask, load_img

IMAGE_SIZE = 256
CROP_SIZE = 256

train_trfm = A.Compose([
    A.RandomCrop(CROP_SIZE, CROP_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomGamma(p=0.5),
    A.GaussNoise(p=0.5),
    A.ISONoise(p=0.5),
    A.Blur(blur_limit=3, p=0.5)
])

val_trfm = A.Compose([
    A.CenterCrop(CROP_SIZE, CROP_SIZE)
])


class RoofTopDataset(D.Dataset):
    def __init__(
            self,
            image_paths=None, mask_paths=None,
            image_list=None, mask_list=None,
            name_list=None,
            transform=None,
            test_mode=False
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_list = image_list
        self.mask_list = mask_list
        self.name_list = name_list
        self.transform = transform
        self.test_mode = test_mode

        if self.image_paths:
            self.len = len(image_paths)
        elif self.image_list:
            self.len = len(image_list)

        self.as_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5],
                        [0.5, 0.5, 0.5]),
        ])

    # get data operation
    def __getitem__(self, index):

        if self.image_paths:
            img = cv2.imread(self.image_paths[index])
        elif self.image_list:
            img = self.image_list[index]

        if not self.test_mode:

            if self.image_paths:
                mask = cv2.imread(self.mask_paths[index], cv2.IMREAD_GRAYSCALE)
            elif self.image_list:
                mask = self.mask_list[index]

            if self.transform is None:
                return self.as_tensor(img), mask[None]
            else:
                augments = self.transform(image=img, mask=mask)
                return self.as_tensor(augments["image"]), augments["mask"][None]  # "None" can add 1st dimension
        else:
            img_name = self.name_list[index]
            return self.as_tensor(img), img_name

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len


def get_train_valid_data(image_folder, mask_folder):
    image_list, mask_list = [], []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        image_paths = glob.glob(image_folder + "/*.png")
        mask_paths = glob.glob(mask_folder + "/*.png")

        for images, masks in executor.map(load_img_mask, image_paths, mask_paths):
            image_list.append(images)
            mask_list.append(masks)

    ds = RoofTopDataset(image_list=image_list, mask_list=mask_list)

    return ds


def get_test_data(image_folder, output_folder=None):
    image_paths = glob.glob(image_folder + "/*.png")
    # image_list = []
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     for image in executor.map(load_img, image_paths):
    #         image_list.append(image)
    name_list = [os.path.basename(img) for img in image_paths]
    if output_folder:
        name_list = [os.path.join(output_folder, img_name) for img_name in name_list]
    test_ds = RoofTopDataset(image_paths=image_paths, name_list=name_list, test_mode=True)

    return test_ds


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from utils.utils import sliding
    import torch.nn.functional as F

    test_ds = get_test_data("./data/test/images")
    loader = D.DataLoader(test_ds, batch_size=1, shuffle=False)

    image = next(iter(loader)).squeeze(0)
    WINDOWS_SIZE = 256
    STEP_SIZE = 256
    h_padding = STEP_SIZE - (image.shape[1] - WINDOWS_SIZE + STEP_SIZE) % STEP_SIZE
    w_padding = STEP_SIZE - (image.shape[2] - WINDOWS_SIZE + STEP_SIZE) % STEP_SIZE

    n_row = (image.shape[1] - WINDOWS_SIZE + h_padding + STEP_SIZE) / STEP_SIZE
    n_col = (image.shape[2] - WINDOWS_SIZE + w_padding + STEP_SIZE) / STEP_SIZE
    padding_image = F.pad(image, (0, w_padding, 0, h_padding))

    print(h_padding, w_padding)
    print(padding_image.shape)
    import time
    import torch

    start = time.time()
    sliding_generator = sliding(padding_image, STEP_SIZE, WINDOWS_SIZE)
    win_stack = torch.stack([win for win in sliding_generator], 0)
    print((time.time() - start))
    print(win_stack.shape)

    print(win_stack[:, :1, :, :].shape)
    from torchvision.utils import make_grid

    print(make_grid(win_stack[:, :1, :, :], nrow=int(n_row), padding=0).shape)

    plt.figure(figsize=(16, 8))
    plt.subplot(121)
    plt.imshow(padding_image[0])
    plt.subplot(122)
    plt.imshow(make_grid(win_stack, nrow=int(n_row), padding=0)[0])
    plt.show()
