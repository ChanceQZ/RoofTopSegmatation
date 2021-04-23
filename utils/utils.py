# -*- coding: utf-8 -*-

"""
@File: utils.py
@Author: Chance (Qian Zhen)
@Description: 
@Date: 2021/04/20
"""
import numpy as np


def sliding(image, step_size, windows_size):
    for row in range(0, image.shape[-2], step_size):
        for col in range(0, image.shape[-1], step_size):
            if image.dim == 4:
                yield image[:, :, row:row + windows_size, col:col + windows_size]
            elif image.dim == 3:
                yield image[:, row:row + windows_size, col:col + windows_size]


if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt
    img_path = "../data/train/images/negative_num3.png"
    img = cv2.imread(img_path)
    print(img.shape)
    crop_img = next(sliding(img, 20, 1024))
    print(crop_img.shape)
    plt.imshow(crop_img[:, :, 0])
    plt.show()