#!usr/bin/env python
# -*- coding:utf-8 -*-

import os
import shutil
from multiprocessing import Pool

import pandas as pd


def galaxy_score(ligand):
    lig_name = ligand.split('.')[0]
    to_ligand = '%s/%s' % (path_name_lig, ligand)
    # log
    path_name_log = '%s/%s_log' % (path_name, lig_name)
    if not os.path.exists(path_name_log):
        os.mkdir(path_name_log)
    # commandline
    cmdline = 'module load anaconda2 &&'
    cmdline += 'cd %s &&' % path_name_log
    cmdline += '/home/xujun/Soft/SCORE_Function/GalaxyDock_BP2/script/calc_energy.py -d \'/home/xujun/Soft/SCORE_Function/GalaxyDock_BP2\' -p %s -l %s' % (
    pre_protein_file, to_ligand)
    os.system(cmdline)


def cal_galaxy():
    global path
    path = r'/home/xujun/Project_2/2_descriptor'
    path = r'/home/xujun/Project_2/0_vina_docking'
    names = ['akt1', 'ampc', 'cp3a4', 'cxcr4', 'gcr', 'hivpr', 'hivrt', 'kif11',
             'KAT2A', 'MAPK1', 'MTORC1', 'VDR', 'PKM2', 'TP53', 'ALDH1', 'ESR1_ant', 'GBA', 'FEN1']
    for name in names:
        global path_name, path_name_lig, pre_protein_file, csv_file
        path_name = '%s/%s' % (path, name)
        protein_file = '%s/%s_protein.pdb' % (path_name, name)
        # create csv file
        csv_file = '%s/galaxy.csv' % path_name
        pd.DataFrame(
            ['name', 'total', 'qq_pl', 'desolv_pl', 'vdw_pl', 'hbond_pl', 'qq_l', 'desolv_l', 'vdw_l', 'hbond_l',
             'DrugScore', 'HMscore', 'PLP_tor']).T.to_csv(csv_file, index=False, header=False)
        # prepare protein
        pre_protein_file = '%s/%s_pg.pdb' % (path_name, name)
        if not os.path.exists(pre_protein_file):
            cmdline = 'cat %s | sed \'/^HETATM/\'d > %s' % (protein_file, pre_protein_file)
            os.system(cmdline)
        # get ligands
        path_name_lig = '%s/dock' % path_name
        ligands = os.listdir(path_name_lig)
        # multi core
        pool = Pool(28)
        pool.map(galaxy_score, ligands)
        pool.close()
        pool.join()
        # merge result
        for ligand in ligands:
            lig_name = ligand.split('.')[0]
            path_name_log = '%s/%s_log' % (path_name, lig_name)
            result = []
            log_file = '%s/energy.log' % path_name_log
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    con = f.readlines()
                for i in range(len(con)):
                    if con[i].startswith('- Total_E'):
                        result = [lig_name] + [con[x].split(':')[-1].strip() for x in range(i, i + 13) if x != i + 1]
            else:
                result = [lig_name]
            pd.DataFrame(result).T.to_csv(csv_file, index=False, header=False, mode='a')
            # remove file
            shutil.rmtree(path_name_log)


if __name__ == '__main__':
    cal_galaxy()
