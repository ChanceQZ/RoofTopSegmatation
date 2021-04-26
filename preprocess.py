# -*- coding: utf-8 -*-

"""
@File: preprocess.py
@Author: Chance (Qian Zhen)
@Description: Training dataset augmentation offline
@Date: 4/26/2021
"""
import cv2
import glob
import time
import numpy as np
import albumentations as A
import concurrent.futures
from multiprocessing import Pool, Manager, cpu_count

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


def transform(image_filename, mask_filename):
    image = cv2.imread(image_filename)
    mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
    # mask = np.where(mask == 1, 255, 0)
    aug_image_list, aug_mask_list = [], []
    for _ in range(AUGMENT_NUM):
        augments = train_trfm(image=image, mask=mask)
        aug_image_list.append(augments["image"])
        aug_mask_list.append(augments["mask"])
    return aug_image_list, aug_mask_list


def save_img(image_filename, img):
    cv2.imwrite(image_filename, img)


def multi_processing_saveimg(
        img_path_list: list,
        img_list: list,
        process_num: int = None
) -> None:
    start = time.time()

    if process_num is None:
        from multiprocessing import cpu_count
        process_num = cpu_count()

    pool = Pool(process_num)
    q = Manager().Queue()

    for img_path, img in zip(img_path_list, img_list):
        pool.apply_async(save_img, args=(img_path, img))
        q.put(img_path)

    pool.close()
    pool.join()


if __name__ == "__main__":
    src_image_path = "./data/train/images"
    src_mask_path = "./data/train/masks"

    aug_image_path = "./data/train_transform/images"
    aug_mask_path = "./data/train_transform/masks"

    aug_image_list, aug_mask_list = [], []

    start = time.time()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        image_files = glob.glob(src_image_path + "/*.png")
        mask_files = glob.glob(src_mask_path + "/*.png")
        for aug_images, aug_masks in executor.map(transform, image_files, mask_files):
            aug_image_list.extend(aug_images)
            aug_mask_list.extend(aug_masks)

    image_path_list = [aug_image_path + "/%d.png" % idx for idx in range(len(aug_image_list))]
    mask_path_list = [aug_mask_path + "/%d.png" % idx for idx in range(len(aug_mask_list))]

    multi_processing_saveimg(image_path_list, aug_image_list, cpu_count())
    multi_processing_saveimg(mask_path_list, aug_mask_list, cpu_count())
    print("Total time cost %.f s" % (time.time() - start))
