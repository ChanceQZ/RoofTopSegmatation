# -*- coding: utf-8 -*-

"""
@File: train.py
@Author: Chance (Qian Zhen)
@Description: 
@Date: 4/20/2021
"""

import time
import numpy as np
import tqdm
import torch
from torch import nn
import torch.utils.data as D
from roottop_dataset import get_train_valid_data
from deeplab_xception import DeepLabv3_plus
from utils.loss_func import SoftDiceLoss

header = r'''
        Train | Valid
Epoch |  Loss |  Loss | Time, m
'''
#          Epoch         metrics            time
raw_line = '{:6d}' + '\u2502{:7.3f}' * 2 + '\u2502{:6.2f}'

bce_fn = nn.BCEWithLogitsLoss()
dice_fn = SoftDiceLoss()


def loss_fn(y_pred, y_true):
    bce = bce_fn(y_pred, y_true)
    dice = dice_fn(y_pred.sigmoid(), y_true)
    return 0.8 * bce + 0.2 * dice


def train(model, train_loader, valid_loader):
    train_params = [{'params': model.get_1x_lr_params(), 'lr': LEARNING_RATE},
                    {'params': model.get_10x_lr_params(), 'lr': LEARNING_RATE * 10}]

    optimizer = torch.optim.AdamW(train_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_loss = 10
    print(header)
    total_train_losses, total_valid_losses = [], []
    for epoch in range(1, EPOCHES + 1):
        epoch_losses = []
        start_time = time.time()
        model.train()
        for image, target in tqdm.tqdm(train_loader):
            image, target = image.to(DEVICE), target.float().to(DEVICE)

            output = model(image)
            loss = loss_fn(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            # print(loss.item())

        vloss = validation(model, valid_loader, loss_fn)
        print(raw_line.format(epoch, np.array(epoch_losses).mean(), vloss,
                              (time.time() - start_time) / 60 ** 1))

        total_train_losses.append(np.array(epoch_losses).mean())
        total_valid_losses.append(vloss)

        if vloss < best_loss:
            best_loss = vloss
            torch.save(model.state_dict(), 'model_weights/model_best.pth')


@torch.no_grad()
def validation(model, loader, loss_fn):
    losses = []
    model.eval()
    for image, target in loader:
        image, target = image.to(DEVICE), target.float().to(DEVICE)
        output = model(image)
        loss = loss_fn(output, target)
        losses.append(loss.item())

    return np.array(losses).mean()


def main():
    image_folder, mask_folder = "./data/train/images", "./data/train/masks"
    train_ds, valid_ds = get_train_valid_data(image_folder, mask_folder, 10)

    print("The image number of training: %d" % len(train_ds))
    print("The image number of training: %d" % len(train_ds))

    train_loader = D.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = D.DataLoader(
        valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = DeepLabv3_plus(N_INPUTCHANNELS, N_CLASS, OUTPUT_STRIDE, pretrained=True, _print=True)

    train(model, train_loader, valid_loader)


if __name__ == "__main__":
    N_INPUTCHANNELS = 3
    N_CLASS = 2
    OUTPUT_STRIDE = 16

    BATCH_SIZE = 16
    EPOCHES = 30
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-3
