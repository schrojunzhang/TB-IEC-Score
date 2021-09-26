#!usr/bin/env python3
# -*- coding:utf-8 -*-
# author=zhang xujun
# time = 2020-06-13
# uesd for modelling with selected features

import pandas as pd
import numpy as np
import sys
from base_data import dataset
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import Normalizer
from sklearn import svm
from sklearn.model_selection import cross_val_score, StratifiedKFold
from hyperopt import fmin, hp, tpe
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, cohen_kappa_score, confusion_matrix

def svm_model(df, *, over_sampling=False, hyper_opt=True, is_dude=True):
    # dataset Ⅰ
    if is_dude:
        #  get x and y
        x = df.iloc[:, 1:-1]
        y = df.iloc[:, -1].astype('int')
        # split dataset
        train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.8, stratify=y, shuffle=True)
    else:  # dataset Ⅱ
        # split dataset
        train_df = df[df[0].str.contains('_T_')]
        test_df = df[df[0].str.contains('_V_')]
        # get x and y
        train_x, train_y = train_df.iloc[:, 1:-1], train_df.iloc[:, -1].astype('int')
        test_x, test_y = test_df.iloc[:, 1:-1], test_df.iloc[:, -1].astype('int')
        # over sampling
    if over_sampling:
        sample = SMOTE()  # instance
        x_sampled, y_sampled = sample.fit_sample(train_x, train_y)  # input data
        train_x = pd.DataFrame(x_sampled)  # trans to Dataframe
        train_y = pd.Series(y_sampled)  # trans to Series
    # datapreprocessing
    scaler = StandardScaler().fit(train_x)
    train_x_a = scaler.transform(train_x)
    test_x_a = scaler.transform(test_x)
    threshold = VarianceThreshold().fit(train_x_a)
    train_x_b = threshold.transform(train_x_a)
    test_x_b = threshold.transform(test_x_a)
    normalizer = Normalizer(norm='l2').fit(train_x_b)
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
    # metrics
    cross_score = cross_val_score(clf, train_x, train_y, cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
                            scoring='f1', n_jobs=10)
    tn, fp, fn, tp = confusion_matrix(test_y, pred).ravel()
    acc = accuracy_score(test_y, pred)
    f1 = f1_score(test_y, pred)
    mcc = matthews_corrcoef(test_y, pred)
    kappa = cohen_kappa_score(test_y, pred)
    return [cross_score, tn, fp, fn, tp, acc, f1, mcc, kappa]

def main(target):  # combination
    # load dataset
    data = dataset(target)  # instance
    lig_name = data.get_item(get_name=True, return_df=True)  # get ligand name
    lig_class = data.get_item(get_class=True, return_df=True)  # get ligand class
    # feature combination
    vdws = ['galaxy', 'goldscore', 'smina']  # Van der Waals interaction
    hbs = ['galaxy', 'goldscore', 'smina']  # Hydrogen bond interaction
    elecs = ['galaxy', 'nnscore']  # Coulomb potential
    lipos = ['chemscore', 'affini', 'xscore', 'xp']  # Hydrophobic energy term
    entropys = ['smo', 'xp']  # Entropy
    clashes = ['chemscore', 'plp']  # Clash effect
    df_polar = data.get_item('affini', energy_type='polar', return_df=True)  # polar
    df_metal = data.get_item('chemscore', energy_type='metal', return_df=True)  # metal interaction
    df_desolv = data.get_item('galaxy', energy_type='desolv', return_df=True)  # Solvation effect
    df_knowledge = data.get_item('dsx', energy_type='knowledge', return_df=True)  # knowledge-based terms
    df_torsion = data.get_item('chemscore', energy_type='torsion', return_df=True)  # torsion terms
    df_non_lipo = data.get_item('smina', energy_type='non_lipo', return_df=True)  # hydrophobic
    df_nn = data.get_item(mode='all', energy_type='nnscore', return_df=True)  # NNscore
    # combine
    for a, vdw in enumerate(vdws):
        for b, hb in enumerate(hbs):
            for c, elec in enumerate(elecs):
                for d, lipo in enumerate(lipos):
                    for e, entropy in enumerate(entropys):
                        for f, clash in enumerate(clashes):
                            df_vdw = data.get_item(vdw, energy_type='vdw', return_df=True)  # Van der Waals interaction
                            df_hb = data.get_item(hb, energy_type='hb', return_df=True)  #  Hydrogen bond interaction
                            df_elec = data.get_item(elec, energy_type='elec', return_df=True)  # Coulomb potential
                            df_lipo = data.get_item(lipo, energy_type='lipo', return_df=True)  # Hydrophobic energy term
                            df_entropy = data.get_item(entropy, energy_type='entropy', return_df=True)  # Entropy
                            df_clash = data.get_item(clash, energy_type='clash', return_df=True)  # Clash effect
                            #  合并数据
                            df_all = pd.concat([lig_name, df_vdw, df_hb, df_elec, df_lipo, df_entropy, df_desolv, df_knowledge,
                                                df_clash, df_metal, df_torsion, df_non_lipo, df_nn, lig_class],
                                               axis=1, ignore_index=True)
                            df_all = pd.DataFrame(df_all, index=None).dropna()  # drop nan
                            # repeat modelling
                            for i in range(10):
                                # 建模
                                cross_score, tn, fp, fn, tp, acc, f1, mcc, kappa = svm_model(df_all, over_sampling=False,
                                                                                             hyper_opt=False, is_dude=True)
                                # print result
                                data.out_report(algo='svm_no_hyper_10', vdw_order=a, hb_order=b, elec_order=c, lipo_order=d, entropy_order=e,
                                                clash_order=f, cross_score=cross_score, tn=tn, fp=fp, fn=fn, tp=tp, acc=acc,
                                                f1=f1, mcc=mcc, kappa=kappa)

if __name__ == '__main__':
    # finished 'ampc', 'akt1', 'cxcr4',
    names = ['hivpr', 'kif11', 'cp3a4', 'gcr', 'hivrt']
    # unfinished：
    for name in names:
        main(name)
