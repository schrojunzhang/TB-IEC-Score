#!usr/bin/env python
# -*- coding:utf-8 -*-
import os
import pandas as pd
from multiprocessing import Pool


# used for cal vina score

def vina_score(ligand):
    import time
    lig_name = ligand.split('.')[0]
    to_ligand = '%s/%s' % (path_name_lig, ligand)
    ligand_pdbqt = '%s/%s.pdbqt' % (path_name_pdbqt, lig_name)
    # log file
    log_file = '%s/%s_vina.txt' % (path_name, lig_name)
    try:
        if os.path.exists(ligand_pdbqt):
            cmdline = 'module purge &&'
            cmdline += 'module load vina &&'
            cmdline += 'vina --receptor %s --ligand %s ' \
                       '--center_x %s --center_y %s --center_z %s --size_x 18.75 --size_y 18.75 --size_z 18.75 --out out.pdbqt --log %s ' \
                       '--score_only --cpu 1' % (protein_pred, ligand_pdbqt, x, y, z, log_file)
            os.system(cmdline)
        else:
            cmdline = 'module purge &&'
            cmdline += 'module load vina &&'
            cmdline += 'prepare_ligand4.py -l %s -o %s -A hydrogens &&' % (to_ligand, ligand_pdbqt)
            cmdline += 'vina --receptor %s --ligand %s ' \
                       '--center_x %s --center_y %s --center_z %s --size_x 18.75 --size_y 18.75 --size_z 18.75 --out out.pdbqt --log %s ' \
                       '--score_only --cpu 1' % (protein_pred, ligand_pdbqt, x, y, z, log_file)
            os.system(cmdline)
    except:
        pass
    # collect score
    sta = time.time()
    result = [lig_name] + [0 for i in range(6)]
    while True:
        end = time.time()
        if os.path.exists(log_file):
            time.sleep(1)
            with open(log_file, 'r') as f:
                con = f.readlines()
            for i in range(len(con)):
                if con[i].startswith('Affinity:') and len(con) - i >= 7:
                    score = con[i].strip().split(':')[1].split('(')[0].strip()
                    gauss1 = con[i + 2].split(':')[-1].strip()
                    gauss2 = con[i + 3].split(':')[-1].strip()
                    repulsion = con[i + 4].split(':')[-1].strip()
                    hydrophobic = con[i + 5].split(':')[-1].strip()
                    hydrogen_bond = con[i + 6].split(':')[-1].strip()
                    rt = (((-0.035579) * float(gauss1) + (-0.005156) * float(gauss2) + (0.84024500000000002) * float(
                        repulsion) + (-0.035069000000000003) * float(hydrophobic) + (-0.58743900000000004) * float(
                        hydrogen_bond)) / float(score) - 1) // 0.058459999999999998
                    result = [lig_name, score, gauss1, gauss2, repulsion, hydrophobic, hydrogen_bond, rt]
            break
        elif end - sta >= 3:
            break
    pd.DataFrame(result).T.to_csv(csv_file, index=None, header=None, mode='a')
    # remove file
    os.remove(log_file)


def cal_vina():
    global path
    path = r'/home/xujun/Project_2/2_descriptor'
    names = ['akt1', 'ampc', 'cp3a4', 'cxcr4', 'gcr', 'hivpr', 'hivrt', 'kif11',
             'KAT2A', 'MAPK1', 'MTORC1', 'VDR', 'PKM2', 'TP53', 'ALDH1', 'ESR1_ant', 'GBA', 'FEN1']
    for name in names:
        global path_name, path_name_lig, protein_pred, csv_file, path_name_pdbqt, x, y, z
        path_name = '%s/%s' % (path, name)
        protein_file = '%s/%s_protein.pdb' % (path_name, name)
        protein_pre = '%s/%s_vina_p.pdb' % (path_name, name)
        protein_pred = '%s/%s_vina_p2.pdbqt' % (path_name, name)
        crystal_src_file = '%s/%s_crystal_ligand.mol2' % (path_name, name)
        crystal_file = '%s/%s_crystal_ligand.pdbqt' % (path_name, name)
        # convert struct
        if not os.path.exists(crystal_file):
            cmdline = 'module purge &&'
            cmdline += 'module load autodock &&'
            cmdline += 'prepare_ligand4.py -l %s -o %s -A hydrogens ' % (crystal_src_file, crystal_file)
            os.system(cmdline)
        # get pocket
        x = os.popen(
            "cat %s | awk '{if ($1==\"ATOM\") print $(NF-6)}' | awk '{x+=$1} END {print x/(NR)}'" % crystal_file).read()
        y = os.popen(
            "cat %s | awk '{if ($1==\"ATOM\") print $(NF-5)}' | awk '{y+=$1} END {print y/(NR)}'" % crystal_file).read()
        z = os.popen(
            "cat %s | awk '{if ($1==\"ATOM\") print $(NF-4)}' | awk '{z+=$1} END {print z/(NR)}'" % crystal_file).read()
        x = float(x.strip())
        y = float(y.strip())
        z = float(z.strip())
        # prepare protein
        if not os.path.exists(protein_pred):
            cmdline = 'cat %s | sed \'/HETATM/\'d > %s &&' % (protein_file, protein_pre)
            cmdline += 'module purge &&'
            cmdline += 'module load autodock &&'
            cmdline += 'prepare_receptor4.py -r %s -o %s -A hydrogens -U nphs_lps_waters_nonstdres &&' % (
                protein_pre, protein_pred)
            cmdline += 'rm  %s' % protein_pre
            os.system(cmdline)
        # write csv file
        csv_file = '%s/vina.csv' % path_name
        pd.DataFrame(
            ['name', 'vina_score', 'gauss1', 'gauss2', 'repulsion', 'hydrophobic', 'HB',
             'rbond']).T.to_csv(
            csv_file, index=None, header=None)
        # get ligands
        path_name_lig = '%s/lig' % path_name
        ligands = os.listdir(path_name_lig)
        # mkdir
        path_name_pdbqt = '%s/lig_pdbqt' % path_name
        if not os.path.exists(path_name_pdbqt):
            os.mkdir(path_name_pdbqt)
        # multicore
        pool = Pool(24)
        pool.map(vina_score, ligands)
        pool.close()
        pool.join()


if __name__ == '__main__':
    cal_vina()
