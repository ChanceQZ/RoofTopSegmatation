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
    N_CLASS = 1
    OUTPUT_STRIDE = 16
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    checkpoints = "model_weights/lr_0.000100/best_model/best_model.pth"
    # main(checkpoints)

    model = DeepLabv3_plus(N_INPUTCHANNELS, N_CLASS, OUTPUT_STRIDE, pretrained=False, _print=True)
    model.load_state_dict(torch.load(checkpoints, map_location=torch.device('cpu')))

    from torchvision import transforms as T
    trfm = T.Compose([
        T.ToPILImage(),
        T.Resize(256),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5]),
    ])
    image = torch.randn(16, 3, 256, 256)
    import cv2
    import matplotlib.pyplot as plt
    img_path = "data/train/images/positive_num35.png"
    img = cv2.imread(img_path)[1200:1200+256, 800:800+256, :]
    model.eval()
    with torch.no_grad():
        pred = model(trfm(img).unsqueeze(0))
    pred_sigmoid = pred.sigmoid().cpu().numpy()
    pred_sigmoid = (pred_sigmoid > 0.5).astype(np.uint8)
    print(np.unique(pred_sigmoid))
    plt.figure(figsize=(16, 8))
    plt.subplot(121)
    plt.imshow(np.where(pred_sigmoid == 1, 255, 0)[0, 0], cmap='gray')
    plt.subplot(122)
    plt.imshow(img)
    plt.show()
