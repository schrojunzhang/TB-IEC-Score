#!usr/bin/env python
# -*- coding:utf-8 -*-
# author = zhang xujun
# time = 2020-07-10

import os
import sys
import csv
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, cohen_kappa_score, matthews_corrcoef, confusion_matrix, accuracy_score

def cal_ef(y_pred, y_true, top_n=0.01):
    df = pd.DataFrame([y_pred, y_true], index=['y_pred', 'y_true']).T
    df.sort_values(by='y_pred', inplace=True, ascending=False)
    total_num = df.shape[0]
    total_ac_num = df[df.y_true == 1].shape[0]
    top_df = df.iloc[:int(total_num*top_n), :]
    top_num = top_df.shape[0]
    top_ac_num = top_df[top_df.y_true == 1].shape[0]
    return (top_ac_num/top_num)/(total_ac_num/total_num)

targets = ['hivpr', 'hivrt']

path = r'./'
for target in targets:
    dst_csv = f'./{target}_score_.csv'
    if not os.path.exists(dst_csv):
        pd.DataFrame(['name', 'cross_score', 'tn', 'fp', 'fn', 'tp', 'acc', 'f1', 'mcc', 'kappa', 'ef_1', 'ef_2', 'ef_5', 'roc']).T.to_csv(dst_csv, index=False, header=False)
    for sf in ['asp', 'chemscore', 'goldscore', 'plp']:
        csv_file = f'{path}/{target}_{sf}_raw.csv'
        df = pd.read_csv(csv_file)
        y_true = df.loc[:, 'class'].values
        for ml in ['svm', 'rf', 'xgb']:
            y_pred = df.loc[:, ml].values
            y_pred_label = [1 if i >= 0.5 else 0 for i in y_pred]
            roc_auc = roc_auc_score(y_true=y_true, y_score=y_pred)
            acc = accuracy_score(y_true, y_pred_label)
            f1 = f1_score(y_true, y_pred_label)
            mcc = matthews_corrcoef(y_true, y_pred_label)
            kappa = cohen_kappa_score(y_true, y_pred_label)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_label).ravel()
            ef_1 = cal_ef(y_pred, y_true, top_n=0.01)
            ef_2 = cal_ef(y_pred, y_true, top_n=0.02)
            ef_5 = cal_ef(y_pred, y_true, top_n=0.05)
            pd.DataFrame([ml, 0, tn, fp, fn, tp, acc, f1, mcc, kappa, ef_1, ef_2, ef_5, roc_auc]).T.to_csv(dst_csv, mode='a', index=False, header=False)




