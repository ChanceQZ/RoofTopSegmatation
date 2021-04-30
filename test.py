# -*- coding: utf-8 -*-

"""
@File: test.py
@Author: Chance (Qian Zhen)
@Description: 
@Date: 2021/04/20
"""

from predict import *
from evaluation import *

def main():
    weights = []
    for weight in weights:
        model = DeepLabv3_plus(
            N_INPUTCHANNELS,
            N_CLASS,
            OUTPUT_STRIDE,
            pretrained=False,
            _print=False
        )

        model.load_state_dict(torch.load(weight, map_location=torch.device(DEVICE)))
        test_ds = get_test_data("./data/test/images")
        test_loader = D.DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=cpu_count()
        )
        ensemble_predict([model], test_loader, ensemble_mode="union")
        print(weight)
        eval_main()

if __name__ == "__main__":
    N_INPUTCHANNELS = 3
    N_CLASS = 1
    OUTPUT_STRIDE = 16
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # DEVICE = "cpu"
    main()