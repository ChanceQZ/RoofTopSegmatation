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
from utils.utils import sliding
import torch.utils.data as D
import torch.nn.functional as F
from torchvision.utils import make_grid

WINDOWS_SIZE = 256
STEP_SIZE = 256


@torch.no_grad()
def predict_image(model, image):
    model = model.to(DEVICE)
    model.eval()

    if image.dim() == 3:
        image = image.unsqueeze(1)
    h_padding = STEP_SIZE - (image.shape[-2] - WINDOWS_SIZE + STEP_SIZE) % STEP_SIZE
    w_padding = STEP_SIZE - (image.shape[-1] - WINDOWS_SIZE + STEP_SIZE) % STEP_SIZE

    n_row = (image.shape[-2] - WINDOWS_SIZE + h_padding + STEP_SIZE) // STEP_SIZE

    padding_image = F.pad(image, (0, w_padding, 0, h_padding))

    sliding_generator = sliding(padding_image, STEP_SIZE, WINDOWS_SIZE)
    win_stack = torch.stack([win for win in sliding_generator], 0)
    # TODO: May overflow GPU memory
    # model(win_stack.to(DEVICE)).shape == (num, 1, WINDOWS_SIZE, WINDOWS_SIZE)
    pred = model(win_stack.to(DEVICE)).squeezed(1).sigmoid().cpu()
    pred = (pred > 0.5).astype(np.uint8)
    # TODO: Only support WINDOWS_SIZE==STEP_SIZE
    pred_merge = make_grid(pred, nrow=n_row, padding=0)[0]
    assert pred_merge.dim() == 2, "dimension of pred_merge is error"
    return pred_merge


def save_image():
    pass


@torch.no_grad()
def ensemble_predict(models, loader, ensemble_mode="voting"):
    for image in tqdm.tqdm(loader):
        results = []
        for model in models:
            result = predict_image(model, image)
            results.append(result)
        result_tensor = torch.stack(results, 0)
        if ensemble_mode == "voting":
            ensemble_result = torch.mode(result_tensor, 0)[0].numpy()
        elif ensemble_mode == "union":
            ensemble_result = torch.any(result_tensor==1, 0).numpy()

def main(checkpoints):
    model = DeepLabv3_plus(N_INPUTCHANNELS, N_CLASS, OUTPUT_STRIDE, pretrained=True, _print=True)
    model.load_state_dict(torch.load(checkpoints))
    models = [model]


    ensemble_predict(model)


if __name__ == "__main__":
    N_INPUTCHANNELS = 3
    N_CLASS = 1
    OUTPUT_STRIDE = 16
    # DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DEVICE = "cpu"
    checkpoints = "model_weights/lr_0.001/best_model/best_model.pth"
    # main(checkpoints)

    model = DeepLabv3_plus(N_INPUTCHANNELS, N_CLASS, OUTPUT_STRIDE, pretrained=False, _print=True)
    model.load_state_dict(torch.load(checkpoints, map_location=torch.device('cpu')))

    import cv2
    import matplotlib.pyplot as plt
    from roottop_dataset import get_train_valid_data

    image_folder = "./data/train/images"
    mask_folder = "./data/train/masks"
    _, ds = get_train_valid_data(image_folder, mask_folder)
    loader = D.DataLoader(ds, batch_size=1, shuffle=True)
    img, mask = next(iter(loader))
    model.eval()
    model = model.to(DEVICE)
    with torch.no_grad():
        pred = model(img.to(DEVICE)).sigmoid().cpu().numpy()[0, 0]
    pred = (pred > 0.5).astype(np.uint8)
    pred = np.where(pred == 1, 255, 0)
    plt.figure(figsize=(24, 8))
    plt.subplot(131)
    plt.imshow(pred, cmap='gray')
    plt.subplot(132)
    plt.imshow(img[0].numpy().transpose(1, 2, 0))
    plt.subplot(133)
    plt.imshow(np.where(mask[0, 0] == 1, 255, 0), cmap='gray')
    plt.show()
