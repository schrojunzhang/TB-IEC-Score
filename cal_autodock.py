#!usr/bin/env python
# -*- coding:utf-8 -*-
import os
import pandas as pd
from multiprocessing import Pool
# used for cal autodock scoring function

def autodock_socre(ligand):
    import time
    lig_name = ligand.split('.')[0]
    to_ligand = '%s/%s' % (path_name_lig, ligand)
    ligand_pdbqt = '%s/%s.pdbqt' % (path_name_pdbqt, lig_name)
    # 计算
    log_file = '%s/%s_autodock.txt' % (path_name, lig_name)  # 分数文件
    if os.path.exists(ligand_pdbqt):
        cmdline = 'module purge &&'
        cmdline += 'module load autodock &&'
        cmdline += 'compute_AutoDock41_score.py -r %s -l %s -o %s' % (
            protein_pred, ligand_pdbqt, log_file)
        os.system(cmdline)
    else:
        cmdline = 'module purge &&'
        cmdline += 'module load autodock &&'
        cmdline += 'prepare_ligand4.py -l %s -o %s -A hydrogens &&' % (to_ligand, ligand_pdbqt)
        cmdline += 'compute_AutoDock41_score.py -r %s -l %s -o %s' % (protein_pred, ligand_pdbqt, log_file)
        os.system(cmdline)
    # 整合结果
    while True:
        if os.path.exists(log_file):
            time.sleep(1)
            break
    with open(log_file, 'r') as f:
        con = f.readlines()
    result = []
    for i in range(len(con)):
        if con[i].endswith('tors\n') and len(con) != 1:
            data = con[i + 1].split(' ')
            result = [lig_name] + list(
                reversed([[data[x] for x in range(len(data)) if data[x] != ''][-i].strip() for i in range(1, 7)]))
    if len(result) > 1:
        try:
            float(result[-1])
        except:
            result = [lig_name]
    else:
        result = [lig_name]
    pd.DataFrame(result).T.to_csv(csv_file, index=None, header=None, mode='a')
    # 删除分数文件
    os.remove(log_file)


def cal_audock():
    global path
    path = r''
    names = ['akt1', 'ampc', 'cp3a4', 'cxcr4', 'gcr', 'hivpr', 'hivrt', 'kif11',
             'KAT2A', 'MAPK1', 'MTORC1', 'VDR', 'PKM2', 'TP53', 'ALDH1', 'ESR1_ant', 'GBA', 'FEN1']
    for name in names:
        global path_name, path_name_lig, protein_pred, csv_file, path_name_pdbqt
        path_name = '%s/%s' % (path, name)
        protein_file = '%s/%s_protein.pdb' % (path_name, name)
        protein_pre = '%s/%s_autodock_p.pdb' % (path_name, name)
        protein_pred = '%s/%s_autodock_p2.pdbqt' % (path_name, name)
        # 处理蛋白
        if not os.path.exists(protein_pred):
            cmdline = 'cat %s | sed \'/HETATM/\'d > %s &&' % (protein_file, protein_pre)
            cmdline += 'module purge &&'
            cmdline += 'module load autodock &&'
            cmdline += 'prepare_receptor4.py -r %s -o %s -A hydrogens -U nphs_lps_waters_nonstdres &&' % (
                protein_pre, protein_pred)
            cmdline += 'rm  %s' % protein_pre
            os.system(cmdline)
        # 写csv文件
        csv_file = '%s/autodock.csv' % path_name
        pd.DataFrame(['name', 'AutoDock4.2Score', 'qq', 'hb', 'vdw', 'dsolv', 'tors']).T.to_csv(csv_file, index=None,
                                                                                                header=None)
        # 获取小分子
        path_name_lig = '%s/lig' % path_name
        ligands = os.listdir(path_name_lig)
        # 创建pdbqt文件夹
        path_name_pdbqt = '%s/lig_pdbqt' % path_name
        if not os.path.exists(path_name_pdbqt):
            os.mkdir(path_name_pdbqt)
        # 调用软件
        pool = Pool(24)
        pool.map(autodock_socre, ligands)
        pool.close()
        pool.join()


if __name__ == '__main__':
    cal_audock()
