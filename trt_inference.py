# -*- coding: utf-8 -*-

"""
@File: trt_inference.py
@Author: Chance (Qian Zhen)
@Description:
@Date: 5/24/2021
"""

import numpy as np
import torch
from torch import nn
from collections import OrderedDict
from deeplab_xception import DeepLabv3_plus
from utils import load_img_mask
from torch2trt import torch2trt
import matplotlib.pyplot as plt
import tqdm


def torch2onnx(model, input_data, onnx_name):
    torch.onnx.export(model, input_data, onnx_name, verbose=True, opset_version=11)

    model = onnx.load(onnx_name)
    model_simp, check = simplify(model)
    onnx.save(model_simp, onnx_name[:-5] + "_simplified.onnx")


def get_data(image_filename, mask_filename):
    image, mask = load_img_mask(image_filename, mask_filename)
    from torchvision import transforms as T
    as_tensor = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5]),
    ])
    return as_tensor(image)[None], mask[None]


def test_acc(torch_model):
    sess_ori = rt.InferenceSession("model_weights/deeplabv3plus.onnx")
    sess_sim = rt.InferenceSession("model_weights/deeplabv3plus_simplified.onnx")
    torch_input, ground_truth = get_data("./data/images/2845.png", "./data/masks/2845.png")
    torch_input = torch_input.cuda()
    onnx_input = {sess_sim.get_inputs()[0].name: torch_input.cpu().numpy()}
    trt_input = torch_input.cpu().numpy()

    with torch.no_grad():
        y_torch = torch_model(torch_input).cpu().numpy()
    y_onnx_ori = sess_ori.run(None, onnx_input)[0]
    y_onnx_sim = sess_sim.run(None, onnx_input)[0]

    model = onnx.load("deeplabv3plus_simplified.onnx")
    engine = backend.prepare(model, device='CUDA:0')
    y_trt = engine.run(trt_input)[0]

    print(np.sum(np.abs(y_torch - y_onnx_ori)))
    print(np.sum(np.abs(y_torch - y_onnx_sim)))
    print(np.sum(np.abs(y_onnx_ori - y_onnx_sim)))
    print(np.sum(np.abs(y_torch - y_trt)))
    print(np.sum(np.abs(y_trt - y_onnx_sim)))

    y_torch = np.where(y_torch >= 0.5, 1, 0)
    y_onnx_ori = np.where(y_onnx_ori >= 0.5, 1, 0)
    y_onnx_sim = np.where(y_onnx_sim >= 0.5, 1, 0)
    y_trt = np.where(y_trt >= 0.5, 1, 0)

    print(np.sum(np.abs(y_torch - y_onnx_ori)))
    print(np.sum(np.abs(y_torch - y_onnx_sim)))
    print(np.sum(np.abs(y_onnx_ori - y_onnx_sim)))
    print(np.sum(np.abs(y_torch - y_trt)))
    print(np.sum(np.abs(y_trt - y_onnx_sim)))

    plt.imshow(np.where(y_trt >= 0.5, 255, 0)[0, 0])
    plt.show()


def test_speed(torch_model):
    torch_input, ground_truth = get_data("./data/images/2845.png", "./data/masks/2845.png")
    torch_input = torch_input.cuda()

    trt_input = torch_input.cpu().numpy()
    import tqdm
    import time

    t1 = time.time()
    with torch.no_grad():
        for _ in tqdm.tqdm(range(100)):
            torch_model(torch_input).cpu().numpy()
    t2 = time.time()
    print("Time cost of inference of pytorch: %.3f" % (t2 - t1))

    model = onnx.load("deeplabv3plus_simplified.onnx")

    engine = backend.prepare(model, device='CUDA:0')
    t3 = time.time()
    for _ in tqdm.tqdm(range(100)):
        engine.run(trt_input)[0]
    print("Time cost of inference of TensorRT: %.3f" % (time.time() - t3))


if __name__ == "__main__":
    model = DeepLabv3_plus(nInputChannels=3, n_classes=1, output_stride=16, pretrained=False, _print=False)
    model.eval()
    model.cuda()
    input_data = torch.rand(1, 3, 384, 384).cuda()
    #
    model_path = "/home/chance/Desktop/best_model.pth"
    state_dict = torch.load(model_path, map_location=torch.device("cuda"))
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    torch_input, ground_truth = get_data("./data/images/2845.png", "./data/masks/2845.png")
    torch_input = torch_input.cuda()



    model_trt = torch2trt(model, [torch_input], fp16_mode=True)
    with torch.no_grad():
        for _ in tqdm.tqdm(range(1000)):
            y = model(torch_input).cpu().numpy()
        for _ in tqdm.tqdm(range(1000)):
            y_trt = model_trt(torch_input).cpu().numpy()
    plt.imshow(np.where(y_trt > 0.5, 255, 0)[0, 0])
    plt.show()
