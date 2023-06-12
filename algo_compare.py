#!usr/bin/env python
# -*- coding:utf-8 -*-
# author = zhang xujun
# time = 2020-07-02
# used for ML algorithms comparison

import os
import sys

import numpy as np
import pandas as pd
from hyperopt import fmin, hp, tpe
from imblearn.over_sampling import SMOTE
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, cohen_kappa_score, confusion_matrix, \
    roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def cal_ef(test_y, pred, *, top=0.01):
    # df
    df = pd.DataFrame(test_y)
    df['pred'] = pred
    # sorted
    df.sort_values(by='pred', inplace=True, ascending=False)
    # remove Isomers
    # data.drop_duplicates(subset='title', keep='first', inplace=True)
    # number of all ligands
    N_total = len(df)
    # number of active ligands
    N_active = len(df[df.iloc[:, 0] == 1])
    # number of top n% ligands
    topb_total = int(N_total * top + 0.5)
    # data of top n%
    topb_data = df.iloc[:topb_total, :]
    # number of active ligands in top n% data
    topb_active = len(topb_data[topb_data.iloc[:, -1] >= 0.5])
    # enrichment factor at top n%
    ef_b = (topb_active / topb_total) / (N_active / N_total)
    return ef_b


def svm_model(df, *, over_sampling=False, hyper_opt=True):
    #  get x and y
    x = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]
    # split dataset
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.8, random_state=42, stratify=y,
                                                        shuffle=True)
    # over sampling
    if over_sampling:
        sample = SMOTE()  # instance
        x_sampled, y_sampled = sample.fit_sample(train_x, train_y)  # input data
        train_x = pd.DataFrame(x_sampled)  # trans to Dataframe
        train_y = pd.Series(y_sampled)  # trans to Series
    # data preprocessing
    scaler = StandardScaler().fit(train_x)  # mean = 0, variance = 1
    train_x_a = scaler.transform(train_x)
    test_x_a = scaler.transform(test_x)
    threshold = VarianceThreshold().fit(train_x_a)
    train_x_b = threshold.transform(train_x_a)
    test_x_b = threshold.transform(test_x_a)
    normalizer = Normalizer(norm='l2').fit(train_x_b)  #
    train_x_c = normalizer.transform(train_x_b)
    test_x_c = normalizer.transform(test_x_b)
    # define data used for training
    train_x = train_x_c
    test_x = test_x_c
    # optimize hyper-parameters
    if hyper_opt:
        def model(hyper_parameter):
            clf = svm.SVC(**hyper_parameter, class_weight='balanced', random_state=42)
            e = cross_val_score(clf, train_x, train_y, cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
                                scoring='f1', n_jobs=10).mean()
            return -e

        hyper_parameter = {'C': hp.uniform('C', 0.1, 10),
                           'gamma': hp.uniform('gamma', 0.001, 1)}  # hyper-parameters
        best = fmin(model, hyper_parameter, algo=tpe.suggest, max_evals=100,
                    rstate=np.random.RandomState(42))  # optimize
        # classifier instance with best hyper-parameters
        clf = svm.SVC(C=best['C'], gamma=best['gamma'], class_weight='balanced', random_state=42,
                      probability=True)
    else:  # no optimization
        # classifier instance with default hyper-parameters
        clf = svm.SVC(class_weight='balanced', random_state=42, probability=True)
    clf.fit(train_x, train_y)  # training
    # test
    pred = clf.predict(test_x)
    pred_pro = clf.predict_proba(test_x)
    pred_pro = [i[1] for i in pred_pro]
    # cal enrichment factor
    ef_1 = cal_ef(test_y, pred_pro, top=0.01)
    ef_2 = cal_ef(test_y, pred_pro, top=0.02)
    ef_5 = cal_ef(test_y, pred_pro, top=0.05)
    # metrics
    cross_score = cross_val_score(clf, train_x, train_y, cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
                                  scoring='f1', n_jobs=10).mean()
    tn, fp, fn, tp = confusion_matrix(test_y, pred).ravel()
    roc_auc = roc_auc_score(y_true=test_y, y_score=pred_pro)
    acc = accuracy_score(test_y, pred)
    f1 = f1_score(test_y, pred)
    mcc = matthews_corrcoef(test_y, pred)
    kappa = cohen_kappa_score(test_y, pred)
    return [ef_1, ef_2, ef_5, cross_score, tn, fp, fn, tp, acc, f1, mcc, kappa, roc_auc, pred_pro]


def rf_model(df, *, over_sampling=False, hyper_opt=True):
    #  get x and y
    x = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]
    # split dataset
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.8, random_state=42, stratify=y,
                                                        shuffle=True)
    # over sampling
    if over_sampling:
        sample = SMOTE()  # instance
        x_sampled, y_sampled = sample.fit_sample(train_x, train_y)  # input data
        train_x = pd.DataFrame(x_sampled)  # trans to Dataframe
        train_y = pd.Series(y_sampled)  # trans to Series
    # data preprocessing
    scaler = StandardScaler().fit(train_x)  # mean = 0, variance = 1
    train_x_a = scaler.transform(train_x)
    test_x_a = scaler.transform(test_x)
    threshold = VarianceThreshold().fit(train_x_a)
    train_x_b = threshold.transform(train_x_a)
    test_x_b = threshold.transform(test_x_a)
    normalizer = Normalizer(norm='l2').fit(train_x_b)  #
    train_x_c = normalizer.transform(train_x_b)
    test_x_c = normalizer.transform(test_x_b)
    # define data used for training
    train_x = train_x_c
    test_x = test_x_c
    # optimize hyper-parameters
    if hyper_opt:
        def model(hyper_parameter):
            clf = RandomForestClassifier(**hyper_parameter, n_jobs=10, random_state=42, class_weight='balanced')
            e = cross_val_score(clf, train_x, train_y, cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
                                scoring='f1', n_jobs=10).mean()
            return -e

        hyper_parameter = {'n_estimators': hp.choice('n_estimators', range(100, 301, 10)),
                           'max_depth': hp.choice('max_depth', range(6, 100)),
                           'max_features': hp.choice('max_features', ['sqrt', 'log2']),
                           'min_samples_leaf': hp.choice('min_samples_leaf', range(3, 10))}  # hyper-parameters
        # hyper-parameter list
        estimators = [i for i in range(100, 301, 10)]
        depth = [i for i in range(6, 100)]
        feature = ['sqrt', 'log2']
        leaf = [i for i in range(3, 10)]
        # optimization
        best = fmin(model, hyper_parameter, algo=tpe.suggest, max_evals=100,
                    rstate=np.random.RandomState(42))
        # classifier instance with best hyper-parameters
        clf = RandomForestClassifier(n_estimators=estimators[best['n_estimators']],
                                     max_depth=depth[best['max_depth']],
                                     max_features=feature[best['max_features']],
                                     min_samples_leaf=leaf[best['min_samples_leaf']],
                                     n_jobs=10, random_state=42, class_weight='balanced')
    else:  # no optimization
        # classifier instance with default hyper-parameters
        clf = RandomForestClassifier(n_jobs=10, random_state=42, class_weight='balanced')
    clf.fit(train_x, train_y)  # training
    # test
    pred = clf.predict(test_x)
    pred_pro = clf.predict_proba(test_x)
    pred_pro = [i[1] for i in pred_pro]
    # cal enrichment factor
    ef_1 = cal_ef(test_y, pred_pro, top=0.01)
    ef_2 = cal_ef(test_y, pred_pro, top=0.02)
    ef_5 = cal_ef(test_y, pred_pro, top=0.05)
    # metrics
    cross_score = cross_val_score(clf, train_x, train_y, cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
                                  scoring='f1', n_jobs=10).mean()
    tn, fp, fn, tp = confusion_matrix(test_y, pred).ravel()
    roc_auc = roc_auc_score(y_true=test_y, y_score=pred_pro)
    acc = accuracy_score(test_y, pred)
    f1 = f1_score(test_y, pred)
    mcc = matthews_corrcoef(test_y, pred)
    kappa = cohen_kappa_score(test_y, pred)
    return [ef_1, ef_2, ef_5, cross_score, tn, fp, fn, tp, acc, f1, mcc, kappa, roc_auc, pred_pro]


def xgb_model(df, *,train_size=0.8, over_sampling=False, hyper_opt=True, return_model=False):
    #  get x and y
    x = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]
    # split dataset
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=train_size, random_state=42, stratify=y,
                                                        shuffle=True)
    # over sampling
    if over_sampling:
        sample = SMOTE()  # instance
        x_sampled, y_sampled = sample.fit_sample(train_x, train_y)  # input data
        train_x = pd.DataFrame(x_sampled)  # trans to Dataframe
        train_y = pd.Series(y_sampled)  # trans to Series
    # data preprocessing
    scaler = StandardScaler().fit(train_x)  # mean = 0, variance = 1
    train_x_a = scaler.transform(train_x)
    test_x_a = scaler.transform(test_x)
    threshold = VarianceThreshold().fit(train_x_a)
    train_x_b = threshold.transform(train_x_a)
    test_x_b = threshold.transform(test_x_a)
    normalizer = Normalizer(norm='l2').fit(train_x_b)  #
    train_x_c = normalizer.transform(train_x_b)
    test_x_c = normalizer.transform(test_x_b)
    # define data used for training
    train_x = train_x_c
    test_x = test_x_c
    # optimize hyper-parameters
    if hyper_opt:
        def model(hyper_parameter):
            clf = XGBClassifier(**hyper_parameter, n_jobs=10, random_state=42)
            e = cross_val_score(clf, train_x, train_y, cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
                                scoring='f1', n_jobs=10).mean()
            return -e

        hyper_parameter = {'n_estimators': hp.choice('n_estimators', range(100, 301, 10)),
                           'max_depth': hp.choice('max_depth', range(3, 11)),
                           'learning_rate': hp.uniform('learning_rate', 0.1, 0.5),
                           'reg_lambda': hp.uniform('reg_lambda', 0.5, 3)}  # hyper-parameters
        # hyper-parameter list
        estimators = [i for i in range(100, 301, 10)]
        depth = [i for i in range(3, 11)]
        # optimization
        best = fmin(model, hyper_parameter, algo=tpe.suggest, max_evals=100,
                    rstate=np.random.RandomState(42))
        # classifier instance with best hyper-parameters
        clf = XGBClassifier(n_estimators=estimators[best['n_estimators']],
                            max_depth=depth[best['max_depth']],
                            learning_rate=best['learning_rate'],
                            reg_lambda=best['reg_lambda'],
                            n_jobs=10, random_state=42)
    else:  # no optimization
        # classifier instance with default hyper-parameters
        clf = XGBClassifier(n_jobs=10, random_state=42)
    clf.fit(train_x, train_y)  # training
    if return_model:
        return clf
    else:
        # test
        pred = clf.predict(test_x)
        pred_pro = clf.predict_proba(test_x)
        pred_pro = [i[1] for i in pred_pro]
        # cal enrichment factor
        ef_1 = cal_ef(test_y, pred_pro, top=0.01)
        ef_2 = cal_ef(test_y, pred_pro, top=0.02)
        ef_5 = cal_ef(test_y, pred_pro, top=0.05)
        # metrics
        cross_score = cross_val_score(clf, train_x, train_y, cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
                                    scoring='f1', n_jobs=10).mean()
        tn, fp, fn, tp = confusion_matrix(test_y, pred).ravel()
        roc_auc = roc_auc_score(y_true=test_y, y_score=pred_pro)
        acc = accuracy_score(test_y, pred)
        f1 = f1_score(test_y, pred)
        mcc = matthews_corrcoef(test_y, pred)
        kappa = cohen_kappa_score(test_y, pred)
        return [ef_1, ef_2, ef_5, cross_score, tn, fp, fn, tp, acc, f1, mcc, kappa, roc_auc, pred_pro]
    
    
def xgb_model_for_tbiecs(x, y=None, *, hyper_opt=True, 
                         model=None,
                         scaler=None,
                         threshold=None,
                         normalizer=None
                         ):
    # data preprocessing
    if scaler is None:
        scaler = StandardScaler().fit(x)  # mean = 0, variance = 1
    train_x = scaler.transform(x)
    if threshold is None:
        threshold = VarianceThreshold().fit(train_x)
    train_x = threshold.transform(train_x)
    if normalizer is None:
        normalizer = Normalizer(norm='l2').fit(train_x)  #
    train_x = normalizer.transform(train_x)
    # optimize hyper-parameters
    if y is not None:
        if hyper_opt:
            def model(hyper_parameter):
                clf = XGBClassifier(**hyper_parameter, n_jobs=10, random_state=42)
                e = cross_val_score(clf, train_x, y, cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
                                    scoring='f1', n_jobs=10).mean()
                return -e

            hyper_parameter = {'n_estimators': hp.choice('n_estimators', range(100, 301, 10)),
                            'max_depth': hp.choice('max_depth', range(3, 11)),
                            'learning_rate': hp.uniform('learning_rate', 0.1, 0.5),
                            'reg_lambda': hp.uniform('reg_lambda', 0.5, 3)}  # hyper-parameters
            # hyper-parameter list
            estimators = [i for i in range(100, 301, 10)]
            depth = [i for i in range(3, 11)]
            # optimization
            best = fmin(model, hyper_parameter, algo=tpe.suggest, max_evals=100,
                        rstate=np.random.RandomState(42))
            # classifier instance with best hyper-parameters
            clf = XGBClassifier(n_estimators=estimators[best['n_estimators']],
                                max_depth=depth[best['max_depth']],
                                learning_rate=best['learning_rate'],
                                reg_lambda=best['reg_lambda'],
                                n_jobs=10, random_state=42)
        else:  # no optimization
            # classifier instance with default hyper-parameters
            clf = XGBClassifier(n_jobs=10, random_state=42)
        # training
        clf.fit(train_x, y)  # training
        return [scaler, threshold, normalizer, clf]
    else:
        clf = model
        # test
        pred = clf.predict(train_x)
        pred_pro = clf.predict_proba(train_x)
        pred_pro = [i[1] for i in pred_pro]
        return pred, pred_pro


def main(name):
    ####################################################################################################################
    # combination
    ####################################################################################################################
    path_src = r''  # path to store feature csv
    path_dst = r''  # path to save result csv
    score_csv = '{}/{}_score.csv'.format(path_dst, name)
    #  create score_csv
    if not os.path.exists(score_csv):
        pd.DataFrame(
            ['name', 'cross_score', 'tn', 'fp', 'fn', 'tp', 'acc', 'f1', 'mcc', 'kappa', 'ef_1', 'ef_2', 'ef_5', 'roc'
             ]).T.to_csv(score_csv, index=False, header=False)
    # load data
    csv_file = '{}/{}_0_1_1_2_0_0.csv'.format(path_src, name)  # feature csv file
    df = pd.read_csv(csv_file, encoding='utf-8').dropna()  # read csv
    # get x and y
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    #  split x and y
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.8, random_state=42, stratify=y,
                                                        shuffle=True)  # 随机按照4：1划分训练集和测试集
    raw_df = pd.concat([test_x.iloc[:, 0], test_y], axis=1)
    raw_data_csv = '{}/{}_raw.csv'.format(path_dst, name)
    # dict
    model_dic = {'svm': svm_model(df, hyper_opt=False),
                 'rf': rf_model(df, hyper_opt=False),
                 'xgb': xgb_model(df, hyper_opt=False)}
    # modelling
    pro_lis = []
    for algo in model_dic:
        ef_1, ef_2, ef_5, cross_score, tn, fp, fn, tp, acc, f1, mcc, kappa, roc_auc, prob = model_dic[
            algo]  # hyperopt
        pro_lis.append([prob, algo])
        pd.DataFrame([algo, cross_score, tn, fp, fn, tp, acc, f1, mcc, kappa, ef_1, ef_2, ef_5, roc_auc]).T.to_csv(
            score_csv, index=False, mode='a', header=False)  # output to csv
    ####################################################################################################################
    # output data
    for prob, algo in pro_lis:
        raw_df[algo] = prob
    raw_df.to_csv(raw_data_csv, encoding='utf-8', index=False)


if __name__ == '__main__':
    name = sys.argv[1]
    # names = ['ampc', 'cxcr4', 'hivpr', 'kif11', 'cp3a4', 'gcr', 'hivrt', 'akt1']
    main(name)
