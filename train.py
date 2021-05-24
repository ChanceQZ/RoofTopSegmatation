# -*- coding: utf-8 -*-

"""
@File: train.py
@Author: Chance (Qian Zhen)
@Description: 
@Date: 4/20/2021
"""

import os
import time
import datetime
import numpy as np
import tqdm
import torch
from torch import nn
import torch.utils.data as D
from roottop_dataset import get_train_valid_data
from deeplab_resnet import DeepLabv3_plus, get_1x_lr_params, get_10x_lr_params
from utils.loss_func import SoftDiceLoss
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter

header = r"""
        Train | Valid
Epoch |  Loss |  Loss | Time, m
"""
#          Epoch         metrics            time
raw_line = "{:6d}" + "\u2502{:7.3f}" * 2 + "\u2502{:6.2f}"

bce_fn = nn.BCEWithLogitsLoss()
dice_fn = SoftDiceLoss()

N_INPUTCHANNELS = 3
N_CLASS = 1
OUTPUT_STRIDE = 16
CHECKPOINTS_SAVE_TIMES = 1  # frequncy of save checkpoints

BATCH_SIZE = 64
EPOCHES = 300
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 0.005
WEIGHT_DECAY = 0.0005

model_svae_path = "./model_weights/lr_{}".format(LEARNING_RATE)

log_folder = "./logs/%s" % (datetime.date.today().strftime('%y-%m-%d'))
writer = SummaryWriter(log_dir=log_folder, flush_secs=60)


def loss_fn(y_pred, y_true):
    bce = bce_fn(y_pred, y_true)
    dice = dice_fn(y_pred.sigmoid(), y_true)
    return 0.8 * bce + 0.2 * dice


def save_loss(total_train_losses, total_valid_losses):
    time_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open("{}/log_epochs{}_lr{}.txt".format(log_folder, EPOCHES, LEARNING_RATE), "w") as f:
        f.write(time_now)
        f.write("\n")
        for idx, (train_loss, valid_loss) in enumerate(zip(total_train_losses, total_valid_losses), 1):
            f.write("epoch:{}, train loss:{}, valid loss:{}\n".format(idx, train_loss, valid_loss))


def train(model, train_loader, valid_loader):
    torch.distributed.init_process_group(backend="nccl")
    local_rank = torch.distributed.get_rank()

    train_params = [{"params": get_1x_lr_params(model), "lr": LEARNING_RATE},
                    {"params": get_10x_lr_params(model), "lr": LEARNING_RATE * 10}]

    optimizer = torch.optim.AdamW(train_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2)
    model = model.to(DEVICE)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # 就这一行
        model = nn.DataParallel(model)

    best_loss = 10
    print(header)
    total_train_losses, total_valid_losses = [], []
    iters = (len(train_loader))
    for epoch in range(EPOCHES):
        epoch_losses = []
        start_time = time.time()
        model.train()
        for idx, (image, target) in tqdm.tqdm(enumerate(train_loader), total=iters):
            image, target = image.to(DEVICE), target.float().to(DEVICE)
            output = model(image)
            loss = loss_fn(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(epoch + idx / iters)

            epoch_losses.append(loss.item())
            # print(loss.item())
            writer.add_scalar("Train_loss", loss.item(), (epoch * iters + idx))

        vloss = validation(model, valid_loader, loss_fn)
        print(raw_line.format(epoch, np.array(epoch_losses).mean(), vloss,
                              (time.time() - start_time) / 60 ** 1))

        total_train_losses.append(np.array(epoch_losses).mean())
        total_valid_losses.append(vloss)

        writer.add_scalar("Valid_loss", total_valid_losses[-1], epoch)

        if epoch % CHECKPOINTS_SAVE_TIMES == 0:
            save_line = "epoch_{:d}_trainloss_{:.4f}_validloss_{:.4f}.pth"
            checkpoints = save_line.format(epoch, total_train_losses[-1], total_valid_losses[-1])
            torch.save(model.state_dict(), os.path.join(model_svae_path, "models", checkpoints))

        if vloss < best_loss:
            best_loss = vloss
            torch.save(model.state_dict(), os.path.join(model_svae_path, "best_model", "best_model_256.pth"))
    save_loss(total_train_losses, total_valid_losses)


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
    train_image_folder, train_mask_folder = "./data/train_transform/images", "./data/train_transform/masks"
    valid_image_folder, valid_mask_folder = "./data/valid/images", "./data/valid/masks"
    train_ds = get_train_valid_data(train_image_folder, train_mask_folder)
    valid_ds = get_train_valid_data(valid_image_folder, valid_mask_folder)
    print("The image number of training: %d" % len(train_ds))
    print("The image number of validation: %d" % len(valid_ds))

    train_loader = D.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = D.DataLoader(
        valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    if not os.path.exists(os.path.join(model_svae_path, "models")):
        os.makedirs(os.path.join(model_svae_path, "models"))
    if not os.path.exists(os.path.join(model_svae_path, "best_model")):
        os.makedirs(os.path.join(model_svae_path, "best_model"))

    model = DeepLabv3_plus(
        nInputChannels=N_INPUTCHANNELS,
        n_classes=N_CLASS,
        output_stride=OUTPUT_STRIDE,
        pretrained=True,
        _print=True
    )
    train(model, train_loader, valid_loader)


if __name__ == "__main__":
    main()
