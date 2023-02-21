#!usr/bin/env python3
# -*- coding:utf-8 -*-
# author = zhang xujun
# time = 2020-06-18
# cal the importance score of energy terms

import os

import pandas as pd

from base_data import dataset


def main():
    # targets
    names = ['akt1', 'ampc', 'cxcr4', 'hivpr', 'kif11', 'cp3a4', 'hivrt', 'gcr']
    # Dataset Ⅰ：'akt1', 'ampc','cxcr4', 'hivpr', 'kif11', 'cp3a4', 'KAT2A',
    # Dataset Ⅱ：'PKM2', 'TP53', 'ALDH1', 'ESR1_ant', 'GBA', 'FEN1', 'MAPK1', 'MTORC1', 'VDR'
    for name in names:
        data = dataset(name)  # instance
        for term in ['vdw', 'hb', 'elec', 'lipo', 'entropy', 'clash']:  # for different energy terms
            # csv_file
            dst_csv = '/home/xujun/Project_2/5_importance/vina_{}.csv'.format(term)
            # cal importance
            importance = data.cal_importance(energy_type=term)  # 计算第一次
            # define columns
            columns = [sf for sf in importance]
            columns_name = ['name'] + columns + ['best']  # rename
            # create csv file
            if not os.path.exists(dst_csv):
                pd.DataFrame(columns_name).T.to_csv(dst_csv, index=False, header=False, mode='w')  # 创建CSV
            for i in range(99):  # repeat 100 times
                new_importance = data.cal_importance(energy_type=term)
                importance = dict(zip(columns, [importance[sf] + new_importance[sf] for sf in columns]))
            avg_importance = [importance[sf] for sf in columns]
            best = [max(importance,
                        key=lambda k: importance[k])]  # get the scoring function achieving the best importance score
            pd.DataFrame([name] + [x / 100 for x in avg_importance] + best).T.to_csv(dst_csv, index=False, header=False,
                                                                                     mode='a')  # output to csv


if __name__ == '__main__':
    main()
