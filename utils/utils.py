# -*- coding: utf-8 -*-

"""
@File: utils.py
@Author: Chance (Qian Zhen)
@Description: 
@Date: 2021/04/20
"""
import numpy as np
import cv2


def sliding(image, step_size, windows_size):
    for row in range(0, image.shape[-2], step_size):
        for col in range(0, image.shape[-1], step_size):
            if image.dim() == 4:
                yield image[:, :, row:row + windows_size, col:col + windows_size]
            elif image.dim() == 3:
                yield image[:, row:row + windows_size, col:col + windows_size]


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
