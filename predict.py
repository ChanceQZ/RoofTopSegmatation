# -*- coding: utf-8 -*-

"""
@File: predict.py
@Author: Chance (Qian Zhen)
@Description: 
@Date: 4/20/2021
"""

import tqdm
import numpy as np
import torch
from utils.metrics import Evaluator
from roottop_dataset import get_test_data
from deeplab_xception import DeepLabv3_plus
from utils.utils import sliding, padding

WINDOWS_SIZE = 512
STEP_SIZE = 512


def image_join(temp, image_arr, h_step, w_step):
    temp = np.zeros_like(temp, dtype=int)
    for h in range(h_step):
        for w in range(w_step):
            temp[h * STEP_SIZE + WINDOWS_SIZE]


@torch.no_grad()
def predict(model, loader):
    model = model.cuda()
    model.eval()
    for image in tqdm.tqdm(loader):
        h_step, w_step = 0, 0
        if image.shape[1] % WINDOWS_SIZE == 0:
            h_step = image.shape[1] // WINDOWS_SIZE
        else:
            h_step = image.shape[1] // WINDOWS_SIZE + 1

        if image.shape[2] % WINDOWS_SIZE == 0:
            w_step = image.shape[2] // WINDOWS_SIZE
        else:
            w_step = image.shape[2] // WINDOWS_SIZE + 1

        padding_image = padding(image, h_step * WINDOWS_SIZE, w_step * WINDOWS_SIZE)

        sliding_generator = sliding(padding_image, STEP_SIZE, WINDOWS_SIZE)
        win_stack = torch.stack([win for win in sliding_generator], 0)
        output = model(win_stack.to(DEVICE))
        pred = np.argmax(output.cpu().numpy(), axis=1)

        image_join(padding_image, pred, h_step, w_step)


def main(checkpoints):
    model = DeepLabv3_plus(N_INPUTCHANNELS, N_CLASS, OUTPUT_STRIDE, pretrained=True, _print=True)
    model.load_state_dict(torch.load(checkpoints))
    predict(model)


if __name__ == "__main__":
    N_INPUTCHANNELS = 3
    N_CLASS = 2
    OUTPUT_STRIDE = 16
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    checkpoints = ""
    main(checkpoints)
