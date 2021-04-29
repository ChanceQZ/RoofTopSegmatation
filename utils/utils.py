# -*- coding: utf-8 -*-

"""
@File: utils.py
@Author: Chance (Qian Zhen)
@Description: 
@Date: 2021/04/20
"""
import psutil
import numpy as np
import cv2
from multiprocessing import Pool, Manager, cpu_count


def sliding(image, step_size, windows_size):
    for row in range(0, image.shape[-2], step_size):
        for col in range(0, image.shape[-1], step_size):
            if image.dim() == 4:
                crop_image = image[:, :, row:row + windows_size, col:col + windows_size]
            elif image.dim() == 3:
                crop_image = image[:, row:row + windows_size, col:col + windows_size]
            if crop_image.shape[-2] == windows_size and crop_image.shape[-1] == windows_size:
                yield crop_image


def fill_hole(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    len_contour = len(contours)
    contour_list = []
    for i in range(len_contour):
        drawing = np.zeros_like(mask, np.uint8)  # create a black image
        img_contour = cv2.drawContours(drawing, contours, i, (255, 255, 255), -1)
        contour_list.append(img_contour)

    out = sum(contour_list)
    return out


def load_mask(mask_filename):
    mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
    return mask


def load_img(image_filename):
    image = cv2.imread(image_filename)
    return image


def load_img_mask(image_filename, mask_filename):
    image = load_img(image_filename)
    mask = load_mask(mask_filename)
    return image, mask


def save_img(image_filename, img):
    cv2.imwrite(image_filename, img)


def multi_processing_saveimg(
        img_path_list: list,
        img_list: list,
        process_num: int = None
) -> None:
    if process_num is None:
        process_num = cpu_count()

    pool = Pool(process_num)
    q = Manager().Queue()

    for img_path, img in zip(img_path_list, img_list):
        pool.apply_async(save_img, args=(img_path, img))
        q.put(img_path)

    pool.close()
    pool.join()


def get_memory_percent():
    virtual_memory = psutil.virtual_memory()
    memory_percent = virtual_memory.percent
    return memory_percent


if __name__ == "__main__":
    import torch
    import torch.nn.functional as F

    STEP_SIZE = 384
    WINDOWS_SIZE = 384
    image = torch.empty(3, 2432, 2432)
    height, width = image.shape[-2], image.shape[-1]
    h_padding = STEP_SIZE - (height - WINDOWS_SIZE + STEP_SIZE) % STEP_SIZE
    w_padding = STEP_SIZE - (width - WINDOWS_SIZE + STEP_SIZE) % STEP_SIZE
    padding_image = F.pad(image, (0, w_padding, 0, h_padding))
    for idx, win in enumerate(sliding(padding_image, STEP_SIZE, WINDOWS_SIZE)):
        print("{} {}".format(idx, win.shape))
