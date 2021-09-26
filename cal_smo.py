#!usr/bin/env python
# -*- coding:utf-8 -*-
import os, pandas as pd
from multiprocessing import Pool
# used for cal SMoG2016

def smog_score(ligand):
    import time
    lig_name = ligand.split('.')[0]
    to_ligand = '%s/%s' % (path_name_lig, ligand)
    # log
    log_file = '%s/%s_smo.txt' % (path_name, lig_name)
    # commandline
    cmdline = 'module load openbabel &&'
    cmdline += 'cd /home/xujun/Soft/SCORE_Function/SMoGxxx &&'
    cmdline += 'chmod +x SMoG2016.exe &&'
    cmdline += '/home/xujun/Soft/SCORE_Function/SMoGxxx/SMoG2016.exe %s %s 0 > %s' % (protein_file, to_ligand, log_file)
    os.system(cmdline)
    # result
    result = [lig_name] + [0 for x in range(5)]
    sta = time.time()
    while True:
        end = time.time()
        if os.path.exists(log_file):
            time.sleep(1)
            with open(log_file, 'r') as f:
                con = f.readlines()
            for i in range(len(con)):
                if con[i].startswith('TOTAL:'):
                    result = [lig_name] + [con[i].split(' ')[x].split(':')[-1].strip() for x in range(5)]
            break
        elif end - sta >= 30:
            break
    pd.DataFrame(result).T.to_csv(csv_file, index=None, header=None, mode='a')
    # remove file
    os.remove(log_file)

def cal_smo():
    global path
    path = r'/home/xujun/Project_2/2_descriptor'
    names = ['akt1', 'ampc', 'cp3a4', 'cxcr4', 'gcr', 'hivpr', 'hivrt', 'kif11',
         'KAT2A', 'MAPK1', 'MTORC1', 'VDR', 'PKM2', 'TP53', 'ALDH1', 'ESR1_ant', 'GBA', 'FEN1']
    names = ['hivrt', 'kif11',
         'KAT2A', 'MAPK1', 'MTORC1', 'VDR', 'PKM2', 'TP53', 'ALDH1', 'ESR1_ant', 'GBA', 'FEN1']
    for name in names:
        global path_name, path_name_lig, protein_file, csv_file
        path_name = '%s/%s' % (path, name)
        protein_file = '%s/%s_protein.pdb' % (path_name, name)
        # write csv file
        csv_file = '%s/smo.csv' % path_name
        pd.DataFrame(['name', 'total', 'SMoG2016_KBP2016', 'SMoG2016_LJP', 'SMoG2016_Rotor', 'SMoG2016_lnMass']).T.to_csv(csv_file, index=None, header=None)
        # get ligands file
        path_name_lig = '%s/lig' % path_name
        ligands = os.listdir(path_name_lig)
        # multi-core
        pool = Pool(24)
        pool.map(smog_score, ligands)
        pool.close()
        pool.join()

if __name__ == '__main__':
    cal_smo()
