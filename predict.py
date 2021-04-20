# -*- coding: utf-8 -*-

"""
@File: predict.py
@Author: Chance (Qian Zhen)
@Description: 
@Date: 4/20/2021
"""

import torch
from roottop_dataset import get_train_valid_data
from deeplab_xception import DeepLabv3_plus

def predict(model):
    pass

def main(checkpoints):
    model = DeepLabv3_plus(N_INPUTCHANNELS, N_CLASS, OUTPUT_STRIDE, pretrained=True, _print=True)
    model.load_state_dict(torch.load(checkpoints))
    predict(model)

if __name__ == "__main__":
    N_INPUTCHANNELS = 3
    N_CLASS = 2
    OUTPUT_STRIDE = 16

    checkpoints = ""
    main(checkpoints)