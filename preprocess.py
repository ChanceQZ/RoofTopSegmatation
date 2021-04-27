# -*- coding: utf-8 -*-

"""
@File: preprocess.py
@Author: Chance (Qian Zhen)
@Description: Training dataset augmentation offline.
              Support up-sampling strategy based on positive and negative
@Date: 4/26/2021
"""
import cv2
import glob
import time
import numpy as np
import albumentations as A
import concurrent.futures
from utils import multi_processing_saveimg

IMAGE_SIZE = 256
CROP_SIZE = 256
AUGMENT_NUM = 50

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
    A.RandomCrop(CROP_SIZE, CROP_SIZE)
])


def transform(image_filename, mask_filename, augment_num):
    image = cv2.imread(image_filename)
    mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
    # Show mask image in software
    # mask = np.where(mask == 1, 255, 0)
    aug_image_list, aug_mask_list = [], []
    for _ in range(augment_num):
        # transfrom need change
        augments = val_trfm(image=image, mask=mask)
        aug_image_list.append(augments["image"])
        aug_mask_list.append(augments["mask"])
    return aug_image_list, aug_mask_list


if __name__ == "__main__":
    POS_SAMPLE_NUM, NEG_SAMPLE_NUM = 1000, 250

    src_image_path = "./data/train/images"
    src_mask_path = "./data/train/masks"

    aug_image_path = "./data/valid/images"
    aug_mask_path = "./data/valid/masks"

    aug_image_list, aug_mask_list = [], []

    start = time.time()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        image_files = glob.glob(src_image_path + "/*.png")
        mask_files = glob.glob(src_mask_path + "/*.png")
        # train_aug_nums = [POS_SAMPLE_NUM if "positive" in name else NEG_SAMPLE_NUM for name in image_files]
        val_aug_nums = [5] * len(image_files)
        for aug_images, aug_masks in executor.map(transform, image_files, mask_files, val_aug_nums):
            aug_image_list.extend(aug_images)
            aug_mask_list.extend(aug_masks)

    image_path_list = [aug_image_path + "/%d.png" % idx for idx in range(len(aug_image_list))]
    mask_path_list = [aug_mask_path + "/%d.png" % idx for idx in range(len(aug_mask_list))]

    multi_processing_saveimg(image_path_list, aug_image_list)
    multi_processing_saveimg(mask_path_list, aug_mask_list)
    print("Total time cost %.f s" % (time.time() - start))
