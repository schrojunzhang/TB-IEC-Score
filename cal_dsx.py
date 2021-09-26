#!usr/bin/env python
# -*- coding:utf-8 -*-
import os
import pandas as pd
from multiprocessing import Pool
# used for cal dsx

def dsx_score(ligand):
    import time
    lig_name = ligand.split('.')[0]
    protein_name = protein_file.split('/')[-1].split('.')[0]
    to_ligand = '%s/%s' % (path_name_lig, ligand)
    result = []
    # 计算
    for i in [0, 2, 3]:
        score_type = ''.join(['-T%s 1 ' % i] + ['-T%s 0 ' % j for j in range(5) if j != i])
        log_file = '/home/xujun/temp/DSX_%s_%s.txt' % (protein_name, lig_name)  # 分数文件
        cmdline = '/home/xujun/Soft/SCORE_Function/dsx090_and_hotspotsx061_linux/linux64/dsx_linux_64.lnx ' \
                  '-D /home/xujun/Soft/SCORE_Function/dsx090_and_hotspotsx061_linux/pdb_pot_0511 ' \
                  '-P %s -L %s ' % (protein_file, to_ligand) + score_type
        os.system(cmdline)
        # 整合结果
        while True:
            if os.path.exists(log_file):
                time.sleep(1)
                break
        with open(log_file, 'r') as f:
            con = f.readlines()

        flag = False
        for i in range(len(con)):
            if con[i].startswith('@RESULTS'):
                result.append(float(con[i + 4].split('|')[3].strip()))
                flag = True
        # 删除分数文件
        os.remove(log_file)
        # 判断计算是否成功
        if not flag:
            break
    if flag:
        result.append(sum(result))
        result.insert(0, lig_name)
    else:
        result = [lig_name]
    pd.DataFrame(result).T.to_csv(csv_file, index=None, header=None, mode='a')


def cal_dsx():
    global path
    path = r'/home/xujun/Project_2/2_descriptor'
    names = ['akt1', 'ampc', 'cp3a4', 'cxcr4', 'gcr', 'hivpr', 'hivrt', 'kif11',
             'KAT2A', 'MAPK1', 'MTORC1', 'VDR', 'PKM2', 'TP53', 'ALDH1', 'ESR1_ant', 'GBA', 'FEN1']
    for name in names:
        global path_name, path_name_lig, protein_file, csv_file
        path_name = '%s/%s' % (path, name)
        protein_file = '%s/%s_protein.pdb' % (path_name, name)
        # 写csv文件
        csv_file = '%s/dsx.csv' % path_name
        pd.DataFrame(['name', 'atom_pairs', 'intra_clashes', 'sas_score', 'total']).T.to_csv(csv_file, index=None,
                                                                                             header=None)
        # 获取小分子
        path_name_lig = '%s/lig' % path_name
        ligands = os.listdir(path_name_lig)
        # 调用软件
        pool = Pool(24)
        pool.map(dsx_score, ligands)
        pool.close()
        pool.join()


if __name__ == '__main__':
    cal_dsx()
