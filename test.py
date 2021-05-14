# -*- coding: utf-8 -*-

"""
@File: test.py
@Author: Chance (Qian Zhen)
@Description: 
@Date: 2021/04/20
"""
import os
import random
import time
import pandas as pd
import itertools
from predict import *
from evaluation import *


def combination_test():
    test_ds = get_test_data("./data/test/images")
    test_loader = D.DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cpu_count()
    )

    weights = sorted(os.listdir("./model_weights/lr_0.005/models"), key=lambda x: int(x.split("_")[1]))[40:]


    for num in range(1, 2):

        performance = {}
        weight_iter = list(itertools.combinations(weights, num))
        random.shuffle(weight_iter)
        weight_combs = list(weight_iter)[:40]

        for weight_comb in weight_combs:
            start = time.time()
            models = []
            weight_name = []
            for weight in weight_comb:
                model = DeepLabv3_plus(
                    N_INPUTCHANNELS,
                    N_CLASS,
                    OUTPUT_STRIDE,
                    pretrained=False,
                    _print=False
                )
                weight_path = os.path.join("./model_weights/lr_0.005/models", weight)
                model.load_state_dict(torch.load(weight_path, map_location=torch.device(DEVICE)))
                models.append(model)
                weight_name.append("_".join(weight.split("_")[:2]))
            ensemble_predict(models, test_loader, ensemble_mode="voting")
            cost_time = time.time() - start
            accuracy, precision, recall, f1 = eval_main()
            print("*".join(weight_name))
            performance["*".join(weight_name)] = [accuracy, precision, recall, f1, cost_time]
        pd.DataFrame(performance).to_csv("ensemble_performance{}.csv".format(num), index=None)


def main():
    weights = os.listdir("./model_weights/lr_0.005/models")
    weights.sort(key=lambda x: int(x.split("_")[1]), reverse=True)
    performance = {}
    for weight in weights:
        if int(weight.split("_")[1]) < 40: continue
        model = DeepLabv3_plus(
            N_INPUTCHANNELS,
            N_CLASS,
            OUTPUT_STRIDE,
            pretrained=False,
            _print=False
        )
        model.load_state_dict(
            torch.load(os.path.join("./model_weights/lr_0.005/models", weight), map_location=torch.device(DEVICE)))
        test_ds = get_test_data("./data/test/images")
        test_loader = D.DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=cpu_count()
        )
        ensemble_predict([model], test_loader, ensemble_mode="union")
        print(weight)
        performance["_".join(weight.split("_")[:2])] = eval_main()


if __name__ == "__main__":
    N_INPUTCHANNELS = 3
    N_CLASS = 1
    OUTPUT_STRIDE = 16
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # DEVICE = "cpu"
    combination_test()
