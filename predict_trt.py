# -*- coding: utf-8 -*-

"""
@File: predict.py
@Author: Chance (Qian Zhen)
@Description: 
@Date: 5/20/2021
"""

import os
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
from torch2trt import torch2trt
from collections import OrderedDict
import argparse


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

        rows = (height - inner_windows_size + h_padding + inner_step_size) // inner_step_size

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

    n_row = len(pred_wins) // rows
    pred = torch.stack(pred_wins, 0)

    pred = (pred > 0.5).type(torch.uint8)

    if fix_flaw:
        pred = pred[:, :, del_padding:-del_padding, del_padding:-del_padding]

    pred_merge = make_grid(pred, nrow=n_row, padding=0)[0]
    if "cuda" in DEVICE:
        torch.cuda.empty_cache()
    assert pred_merge.dim() == 2, "dimension of pred_merge is error"

    return pred_merge[:height, :width]


@torch.no_grad()
def ensemble_predict(models, loader, ensemble_mode="voting"):
    for image, output_path in tqdm.tqdm(loader):
        if np.all(image.numpy() == -1):
            cv2.imwrite(output_path[0], np.zeros(image.numpy().shape[-2:]))
            continue

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

        cv2.imwrite(output_path[0], ensemble_result)


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
        input_data = torch.rand(1, 3, 384, 384).to(DEVICE)
        model_trt = torch2trt(model, [input_data], fp16_mode=True)

        models.append(model_trt)

    if args.batch_folder is None:
        input_folder_list = args.input_folder.split(",")
        output_folder_list = args.output_folder.split(",")
    else:
        input_folder_list, output_folder_list = [], []
        with open(args.batch_folder) as f:
            for row in f.readlines():
                row = row.strip()
                input_folder_list.append(row.split(",")[0])
                output_folder_list.append(row.split(",")[1])

    ds_list = []
    for input_folder, output_folder in zip(input_folder_list, output_folder_list):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        ds = get_test_data(input_folder, output_folder)
        ds_list.append(ds)
    test_ds = D.ConcatDataset(ds_list)

    test_loader = D.DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    ensemble_predict(models, test_loader, ensemble_mode="union")


if __name__ == "__main__":
    WINDOWS_SIZE = 384
    STEP_SIZE = 384
    N_INPUTCHANNELS = 3
    N_CLASS = 1
    OUTPUT_STRIDE = 16

    parser = argparse.ArgumentParser()
    parser.description = 'please enter two parameters a and b ...'
    parser.add_argument("--input_folder", help="input folder path", type=str, default="test_input_folder")
    parser.add_argument("--output_folder", help="output folder path", type=str, default="test_output_folder")
    parser.add_argument("--batch_folder", help="batch folder path", type=str, default=None)
    parser.add_argument("--device", help="device", type=str, default="")
    args = parser.parse_args()

    # if args.device == "":
    #     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # else:
    #     DEVICE = args.device

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device != "":
        gpu_idx = int(args.device.split(":")[1])
        torch.cuda.set_device(gpu_idx)

    pred_main()
