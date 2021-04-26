# -*- coding: utf-8 -*-

"""
@File: predict.py
@Author: Chance (Qian Zhen)
@Description: 
@Date: 4/20/2021
"""

import utils
import tqdm
import cv2
import numpy as np
import torch
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

    height, width = image.shape[-2], image.shape[-1]

    if image.dim() == 4:
        image = image.squeeze(0)
    h_padding = STEP_SIZE - (height - WINDOWS_SIZE + STEP_SIZE) % STEP_SIZE
    w_padding = STEP_SIZE - (width - WINDOWS_SIZE + STEP_SIZE) % STEP_SIZE

    n_row = (height - WINDOWS_SIZE + h_padding + STEP_SIZE) // STEP_SIZE

    padding_image = F.pad(image, (0, w_padding, 0, h_padding))
    sliding_generator = sliding(padding_image, STEP_SIZE, WINDOWS_SIZE)
    win_stack = torch.stack([win for win in sliding_generator], 0)
    # TODO: May overflow GPU memory
    # model(win_stack.to(DEVICE)).shape == (num, 1, WINDOWS_SIZE, WINDOWS_SIZE)
    pred = model(win_stack.to(DEVICE)).sigmoid().cpu()
    pred = (pred > 0.5).type(torch.uint8)
    # TODO: Only support WINDOWS_SIZE==STEP_SIZE
    pred_merge = make_grid(pred, nrow=n_row, padding=0)[0]

    assert pred_merge.dim() == 2, "dimension of pred_merge is error"

    return pred_merge[:height, :width]


def save_image():
    pass


@torch.no_grad()
def ensemble_predict(models, loader, ensemble_mode="voting"):
    for image, img_name in tqdm.tqdm(loader):
        results = []
        for model in models:
            result = predict_image(model, image)
            results.append(result)
        result_tensor = torch.stack(results, 0)
        ensemble_result = None
        if ensemble_mode == "voting":
            ensemble_result = torch.mode(result_tensor, 0)[0].numpy()
        elif ensemble_mode == "union":
            ensemble_result = torch.any(result_tensor == 1, 0).numpy().astype(np.uint8)
        # ensemble_result = np.where(ensemble_result == 1, 255, 0)
        cv2.imwrite("./data/test/ensemble_predict/%s" % img_name, ensemble_result)
        # print("./data/test/ensemble_predict/%s" % img_name)


def main():
    import json
    ensamble_config = "ensemble_config.json"
    with open(ensamble_config) as f:
        weights = json.load(f)

    models = []
    for weight in weights.values():
        model = DeepLabv3_plus(N_INPUTCHANNELS, N_CLASS, OUTPUT_STRIDE, pretrained=False, _print=False)
        model.load_state_dict(torch.load(weight, map_location=torch.device(DEVICE)))
        models.append(model)

    test_ds = get_test_data("./data/test/images")
    test_loader = D.DataLoader(test_ds, batch_size=1, shuffle=False)

    ensemble_predict(models, test_loader, ensemble_mode="union")


if __name__ == "__main__":

    N_INPUTCHANNELS = 3
    N_CLASS = 1
    OUTPUT_STRIDE = 16
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DEVICE = "cpu"
    main()
