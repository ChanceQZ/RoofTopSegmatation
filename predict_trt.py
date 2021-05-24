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
from utils import sliding, fill_hole
import torch.utils.data as D
import torch.nn.functional as F
from torchvision.utils import make_grid
from multiprocessing import cpu_count
from utils import multi_processing_saveimg, get_memory_percent
from torch2trt import torch2trt
from collections import OrderedDict

WINDOWS_SIZE = 384
STEP_SIZE = 384
N_INPUTCHANNELS = 3
N_CLASS = 1
OUTPUT_STRIDE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def predict_image(model, image, fix_flaw=False):
    """
    Prediction in sequence
    """
    height, width = image.shape[-2], image.shape[-1]

    if image.dim() == 4:
        image = image.squeeze(0)

    if fix_flaw:
        inner_windows_size = 256
        inner_step_size = 256

        h_padding = inner_step_size - (height - inner_windows_size + inner_step_size) % inner_step_size
        w_padding = inner_step_size - (width - inner_windows_size + inner_step_size) % inner_step_size

        n_row = (height - inner_windows_size + h_padding + inner_step_size) // inner_step_size

        del_padding = (WINDOWS_SIZE - inner_windows_size) // 2
        padding_image = F.pad(image, (0, w_padding, 0, h_padding))
        padding_image = F.pad(padding_image, tuple([del_padding] * 4))
        sliding_generator = sliding(padding_image, STEP_SIZE - del_padding * 2, WINDOWS_SIZE)
    else:
        h_padding = STEP_SIZE - (height - WINDOWS_SIZE + STEP_SIZE) % STEP_SIZE
        w_padding = STEP_SIZE - (width - WINDOWS_SIZE + STEP_SIZE) % STEP_SIZE

        n_row = (height - WINDOWS_SIZE + h_padding + STEP_SIZE) // STEP_SIZE

        padding_image = F.pad(image, (0, w_padding, 0, h_padding))
        sliding_generator = sliding(padding_image, STEP_SIZE, WINDOWS_SIZE)

    pred_wins = []
    for win in sliding_generator:
        pred_win = model(win.unsqueeze(0).to(DEVICE)).sigmoid().cpu()
        pred_wins.append(pred_win.squeeze(0))
    pred = torch.stack(pred_wins, 0)


    pred = (pred > 0.5).type(torch.uint8)

    if fix_flaw:
        pred = pred[:, :, del_padding:-del_padding, del_padding:-del_padding]

    pred_merge = make_grid(pred, nrow=n_row, padding=0)[0]
    torch.cuda.empty_cache()
    assert pred_merge.dim() == 2, "dimension of pred_merge is error"

    return pred_merge[:height, :width]


@torch.no_grad()
def ensemble_predict(models, loader, ensemble_mode="voting"):
    image_list, image_path_list = [], []
    for image, image_name in tqdm.tqdm(loader):
        results = []
        for model in models:
            result = predict_image(model, image, fix_flaw=True)
            results.append(result)
        result_tensor = torch.stack(results, 0)
        ensemble_result = None
        if ensemble_mode == "voting":
            ensemble_result = torch.mode(result_tensor, 0)[0].numpy()
        elif ensemble_mode == "union":
            ensemble_result = torch.any(result_tensor == 1, 0).numpy().astype(np.uint8)

        ensemble_result = np.where(ensemble_result == 1, 255, 0)

        cv2.imwrite("/home/chance/Windows_Disks/G/RoofTopSegmatation/data/test_180/predict/%s" % image_name, ensemble_result)
        # print("./data/test/union_ensemble_predict/%s" % img_name)
    #     image_list.append(ensemble_result)
    #     image_path_list.append("./data/test/union_ensemble_predict/%s" % image_name)
    #
    #     if get_memory_percent() > 90:
    #         multi_processing_saveimg(image_path_list, image_list)
    #         image_list, image_path_list = [], []
    # multi_processing_saveimg(image_path_list, image_list)


def pred_main():
    import json
    ensamble_config = "ensemble_config.json"
    with open(ensamble_config) as f:
        weights = json.load(f)

    models = []
    for weight in weights.values():
        model = DeepLabv3_plus(
            N_INPUTCHANNELS,
            N_CLASS,
            OUTPUT_STRIDE,
            pretrained=False,
            _print=False
        )

        state_dict = torch.load(weight, map_location=torch.device(DEVICE))
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

        model = model.to(DEVICE)
        model.eval()
        input_data = torch.rand(1, 3, 384, 384).cuda()
        model_trt = torch2trt(model, [input_data], fp16_mode=True)

        models.append(model_trt)

    test_ds = get_test_data("/home/chance/Windows_Disks/G/RoofTopSegmatation/data/test_180/images")
    test_loader = D.DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cpu_count()
    )

    ensemble_predict(models, test_loader, ensemble_mode="voting")


if __name__ == "__main__":
    # DEVICE = "cpu"
    pred_main()
