# -*- coding: utf-8 -*-

"""
@File: evaluation.py
@Author: Chance (Qian Zhen)
@Description: 
@Date: 2021/04/20
"""

import os
import glob
from tqdm import tqdm
from PIL import Image
import numpy as np
import concurrent.futures
from utils import load_mask
from sklearn.metrics import confusion_matrix


def confusion_mat_eval(truth_image, pred_image):
    truth = truth_image.reshape((-1)).tolist()
    predict = pred_image.reshape((-1)).tolist()
    con_mat = confusion_matrix(truth, predict)

    try:
        TN, FP, FN, TP = con_mat.ravel()
    except:
        TN = con_mat.ravel()[0]
        FP, FN, TP = 0, 0, 0

    return TN, FP, FN, TP


def evaluate(truth_image_list, pred_image_list):
    TN_total, FP_total, FN_total, TP_total = 0, 0, 0, 0
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for TN, FP, FN, TP in tqdm(executor.map(confusion_mat_eval,
                                                truth_image_list,
                                                pred_image_list),
                                   total=len(truth_image_list)):
            TN_total += TN
            FP_total += FP
            FN_total += FN
            TP_total += TP

    print()

    accuracy = (TP_total + TN_total) / (TP_total + TN_total + FP_total + FN_total)
    # specificity = TN_total / (TN_total + FP_total)
    precision = TP_total / (TP_total + FP_total)
    recall = TP_total / (TP_total + FN_total)
    f1 = (2 * precision * recall) / (precision + recall)

    print("toatl_accuracy: %f" % accuracy)
    print("total_precision: %f" % precision)
    print("total_recall: %f" % recall)
    # print("total_specificity: %f" %specificity)
    print("total_fmeasure: %f" % f1)
    return accuracy, precision, recall, f1


def eval_main():
    # truth_folder = "./data/test_180/labels"
    # pred_floder = "./data/test_180/predict"
    truth_folder = "/home/chance/Windows_Disks/G/Project/RooftopUnderstanding/Data/dataset/old/split/valid/mask_1"
    pred_folder = "/home/chance/Windows_Disks/G/Project/RooftopUnderstanding/Data/dataset/old/split/valid/predict_DeeplabV3p_new"
    # truth_path_list = glob.gloob(truth_folder + "/*.tif")
    # pred_path_list = glob.glob(pred_floder + "/*.tif")

    # truth_folder = "./data/test_180/labels"
    # pred_folder = "./data/test_180/predict"
    # truth_path_list = []
    # pred_path_list = []
    # for pred_path in glob.glob(pred_floder + "/*.tif"):
    #     if os.path.exists(os.path.join(truth_folder, os.path.basename(pred_path))):
    #         truth_path_list.append(os.path.join(truth_folder, os.path.basename(pred_path)))
    #         pred_path_list.append(os.path.join(pred_floder, os.path.basename(pred_path)))
    #

    truth_path_list = sorted(glob.glob(truth_folder + "/*.png"))
    pred_path_list = sorted(glob.glob(pred_folder + "/*.png"))
    # truth_path_list = [truth_path for truth_path in truth_path_list if "lev1" in truth_path]
    # pred_path_list = [pred_path for pred_path in pred_path_list if "lev1" in pred_path]



    truth_image_list, pred_image_list = [], []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for truth_image in executor.map(load_mask, truth_path_list):
            # truth_image = np.where(truth_image == 1, 1, 0)
            truth_image_list.append(truth_image)

        for pred_image in executor.map(load_mask, pred_path_list):
            pred_image = np.where(pred_image == 255, 1, 0)
            pred_image_list.append(pred_image)
    return evaluate(truth_image_list, pred_image_list)



if __name__ == "__main__":
    eval_main()