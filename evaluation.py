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
from sklearn.metrics import confusion_matrix

img_truth_folder = "./data/test/masks"
img_pred_floder = "./data/test/ensemble_predict"
img_truth_list = glob.glob(img_truth_folder + "/*.png")
img_pred_list = glob.glob(img_pred_floder + "/*.png")

acc_list, pre_list, recall_list, f1_list = [], [], [], []

TN_total, FP_total, FN_total, TP_total = 0, 0, 0, 0
for img_truth, img_pred in tqdm(zip(img_truth_list, img_pred_list)):
    print(os.path.basename(img_truth))

    truth = np.array(Image.open(img_truth).convert('L')).reshape((-1)).tolist()
    predict = np.array(Image.open(img_pred).convert('L')).reshape((-1)).tolist()

    con_mat = confusion_matrix(truth, predict)

    try:
        TN, FP, FN, TP = con_mat.ravel()
    except:
        TN = con_mat.ravel()[0]
        FP, FN, TP = 0, 0, 0

    TN_total += TN
    FP_total += FP
    FN_total += FN
    TP_total += TP


print()
accuracy = (TP_total + TN_total) / (TP_total + TN_total + FP_total + FN_total)
specificity = TN_total / (TN_total + FP_total)
precision = TP_total / (TP_total + FP_total)
recall = TP_total / (TP_total + FN_total)
f1 = (2 * precision * recall) / (precision + recall)

print("toatl_accuracy: %f" % accuracy)
print("total_precision: %f" % precision)
print("total_recall: %f" % recall)
#print("total_specificity: %f" %specificity)
print("total_fmeasure: %f" % f1)