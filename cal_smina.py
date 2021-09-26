#!usr/bin/env python
# -*- coding:utf-8 -*-
import os
import pandas as pd
from multiprocessing import Pool
# used for cal smina

def smina_score(ligand):
    import time
    lig_name = ligand.split('.')[0]
    to_ligand = '%s/%s' % (path_name_lig, ligand)
    ligand_pdbqt = '%s/%s.pdbqt' % (path_name_pdbqt, lig_name)
    # log
    log_file = '%s/%s_smina.txt' % (path_name, lig_name)
    if os.path.exists(ligand_pdbqt):
        cmdline = '/home/xujun/Soft/SCORE_Function/smina/smina --receptor %s --ligand %s ' \
                  '--center_x %s --center_y %s --center_z %s --size_x 18.75 --size_y 18.75 --size_z 18.75 --log %s ' \
                  '--custom_scoring /home/xujun/Soft/SCORE_Function/smina/total.score ' \
                  '--score_only --cpu 1' % (protein_pred, ligand_pdbqt, x, y, z, log_file)
        os.system(cmdline)
    else:
        cmdline = 'module purge &&'
        cmdline += 'module load autodock &&'
        cmdline += 'prepare_ligand4.py -l %s -o %s -A hydrogens &&' % (to_ligand, ligand_pdbqt)
        cmdline = '/home/xujun/Soft/SCORE_Function/smina/smina --receptor %s --ligand %s ' \
                  '--center_x %s --center_y %s --center_z %s --size_x 18.75 --size_y 18.75 --size_z 18.75 --log %s ' \
                  '--custom_scoring /home/xujun/Soft/SCORE_Function/smina/total.score ' \
                  '--score_only --cpu 1' % (protein_pred, ligand_pdbqt, x, y, z, log_file)
        os.system(cmdline)
    # collect file
    while True:
        if os.path.exists(log_file):
            time.sleep(1)
            break
    with open(log_file, 'r') as f:
        con = f.readlines()
    result = [lig_name] + [0 for i in range(81)]
    for i in range(len(con)):
        if con[i].startswith('Term values, before weighting:'):
            result = [lig_name] + [x.strip() for x in con[i + 1].lstrip('##  ').split(' ')]
    pd.DataFrame(result).T.to_csv(csv_file, index=None, header=None, mode='a')
    # 删除分数文件
    os.remove(log_file)


def cal_smina():
    global path
    path = r'/home/xujun/Project_2/2_descriptor'
    names = ['ciap']
    for name in names:
        global path_name, path_name_lig, protein_pred, csv_file, path_name_pdbqt, x, y, z
        path_name = '%s/%s' % (path, name)
        protein_file = '%s/%s_protein.pdb' % (path_name, name)
        protein_pre = '%s/%s_smina_p.pdb' % (path_name, name)
        protein_pred = '%s/%s_smina_p2.pdbqt' % (path_name, name)
        crystal_src_file = '%s/%s_crystal_ligand.mol2' % (path_name, name)
        crystal_file = '%s/%s_crystal_ligand.pdbqt' % (path_name, name)
        # move file and convert struct
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
        csv_file = '%s/smina.csv' % path_name
        pd.DataFrame(
            ['name', 'gauss(o=0,_w=0.3,_c=8)', 'gauss(o=0.5,_w=0.3,_c=8)', 'gauss(o=1,_w=0.3,_c=8)',
             'gauss(o=1.5,_w=0.3,_c=8)', 'gauss(o=2,_w=0.3,_c=8)', 'gauss(o=2.5,_w=0.3,_c=8)', 'gauss(o=0,_w=0.5,_c=8)'
                , 'gauss(o=1,_w=0.5,_c=8)', 'gauss(o=2,_w=0.5,_c=8)', 'gauss(o=0,_w=0.7,_c=8)',
             'gauss(o=1,_w=0.7,_c=8)',
             'gauss(o=2,_w=0.7,_c=8)', 'gauss(o=0,_w=0.9,_c=8)', 'gauss(o=1,_w=0.9,_c=8)', 'gauss(o=2,_w=0.9,_c=8)',
             'gauss(o=3,_w=0.9,_c=8)', 'gauss(o=0,_w=1.5,_c=8)', 'gauss(o=1,_w=1.5,_c=8)', 'gauss(o=2,_w=1.5,_c=8)',
             'gauss(o=3,_w=1.5,_c=8)', 'gauss(o=4,_w=1.5,_c=8)', 'gauss(o=0,_w=2,_c=8)', 'gauss(o=1,_w=2,_c=8)',
             'gauss(o=2,_w=2,_c=8)', 'gauss(o=3,_w=2,_c=8)', 'gauss(o=4,_w=2,_c=8)', 'gauss(o=0,_w=3,_c=8)',
             'gauss(o=1,_w=3,_c=8)', 'gauss(o=2,_w=3,_c=8)', 'gauss(o=3,_w=3,_c=8)', 'gauss(o=4,_w=3,_c=8)',
             'repulsion(o=0.4,_c=8)', 'repulsion(o=0.2,_c=8)', 'repulsion(o=0,_c=8)', 'repulsion(o=-0.2,_c=8)',
             'repulsion(o=-0.4,_c=8)', 'repulsion(o=-0.6,_c=8)', 'repulsion(o=-0.8,_c=8)', 'repulsion(o=-1,_c=8)',
             'hydrophobic(g=0.5,_b=1.5,_c=8)', 'hydrophobic(g=0.5,_b=1,_c=8)', 'hydrophobic(g=0.5,_b=2,_c=8)',
             'hydrophobic(g=0.5,_b=3,_c=8)', 'non_hydrophobic(g=0.5,_b=1.5,_c=8)', 'vdw(i=4,_j=8,_s=0,_^=100,_c=8)',
             'vdw(i=6,_j=12,_s=1,_^=100,_c=8)', 'non_dir_h_bond(g=-0.7,_b=0,_c=8)', 'non_dir_h_bond(g=-0.7,_b=0.2,_c=8)'
                , 'non_dir_h_bond(g=-0.7,_b=0.5,_c=8)', 'non_dir_h_bond(g=-1,_b=0,_c=8)',
             'non_dir_h_bond(g=-1,_b=0.2,_c=8)',
             'non_dir_h_bond(g=-1,_b=0.5,_c=8)', 'non_dir_h_bond(g=-1.3,_b=0,_c=8)',
             'non_dir_h_bond(g=-1.3,_b=0.2,_c=8)',
             'non_dir_h_bond(g=-1.3,_b=0.5,_c=8)', 'non_dir_anti_h_bond_quadratic(o=0,_c=8)',
             'non_dir_anti_h_bond_quadratic(o=0.5,_c=8)', 'non_dir_anti_h_bond_quadratic(o=1,_c=8)',
             'donor_donor_quadratic(o=0,_c=8)', 'donor_donor_quadratic(o=0.5,_c=8)', 'donor_donor_quadratic(o=1,_c=8)',
             'acceptor_acceptor_quadratic(o=0,_c=8)', 'acceptor_acceptor_quadratic(o=0.5,_c=8)',
             'acceptor_acceptor_quadratic(o=1,_c=8)', 'non_dir_h_bond_lj(o=-0.7,_^=100,_c=8)',
             'non_dir_h_bond_lj(o=-1,_^=100,_c=8)', 'non_dir_h_bond_lj(o=-1.3,_^=100,_c=8)',
             'ad4_solvation(d-sigma=3.6,_s/q=0.01097,_c=8)', 'ad4_solvation(d-sigma=3.6,_s/q=0,_c=8)',
             'electrostatic(i=1,_^=100,_c=8)', 'electrostatic(i=2,_^=100,_c=8)', 'num_tors_div', 'num_tors_div_simple',
             'num_heavy_atoms_div', 'num_heavy_atoms', 'num_tors_add', 'num_tors_sqr', 'num_tors_sqrt',
             'num_hydrophobic_atoms', 'ligand_length', 'num_ligands']).T.to_csv(
            csv_file, index=None, header=None)
        # get ligands
        path_name_lig = '%s/lig' % path_name
        ligands = os.listdir(path_name_lig)
        # create pdbqt file folder
        path_name_pdbqt = '%s/lig_pdbqt' % path_name
        if not os.path.exists(path_name_pdbqt):
            os.mkdir(path_name_pdbqt)
        # 调用软件
        pool = Pool(24)
        pool.map(smina_score, ligands)
        pool.close()
        pool.join()


if __name__ == '__main__':
    cal_smina()
