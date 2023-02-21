#!usr/bin/env python
# -*- coding:utf-8 -*-
import os, pandas as pd
from multiprocessing import Pool
# used for cal affiscore

def affini(ligand):
    import time
    lig_name = ligand.split('.')[0]
    to_ligand = '%s/%s' % (path_name_lig, ligand)
    lig_temp = '%s/%s_temp.mol2' % (path_name, lig_name)

    with open(to_ligand, 'r') as f:
        lig_data = f.read()
    if '@<TRIPOS>DICT' in lig_data:
        cmdline = 'cat %s | sed \'/@<TRIPOS>DICT/,/@<TRIPOS>ATOM/c @<TRIPOS>ATOM\' > %s ' % (to_ligand, lig_temp)
        os.system(cmdline)
    else:
        cmdline = 'cp %s %s' % (to_ligand, lig_temp)
        os.system(cmdline)
    newline = []
    with open(lig_temp, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if len(line.split()) >= 3:
            if line.split()[1] == 'OXT':
                newline.append(line[:47] + 'O.co2' + line[52:])
            else:
                newline.append(line)
        else:
            newline.append(line)
    with open(lig_temp, 'w') as f:
        f.write(''.join(newline))
    # log
    log_file = '%s/%s_log.txt' % (path_name, lig_name)
    # command line
    cmdline = 'export SLIDE_DIR=/home/xujun/Soft/Score_Function/SLIDE/SLIDE-master &&'
    cmdline += '${SLIDE_DIR}/bin/slide_score -p %s -l %s > %s' % (pre_protein_file, lig_temp, log_file)
    os.system(cmdline)
    # merge result
    while True:
        if os.path.exists(log_file):
            time.sleep(1)
            break
    with open(log_file, 'r') as f:
        con = f.readlines()
    result = [lig_name] + [0 for x in range(13)]
    for i in range(len(con)):
        if con[i].startswith('  1                  2') and not con[i+1].startswith('WARNING:'):
            result = [lig_name] + [con[i + 1].split()[-j].strip('[').strip(']').strip() for j in reversed(range(1, 16, ))
                             if j !=
                             12 and j != 14]
    pd.DataFrame(result).T.to_csv(csv_file, index=None, header=None, mode='a')
    # remove file
    os.remove(lig_temp)
    os.remove(log_file)

def cal_aff():
    global path
    path = r'/home/xujun/Project_2/2_descriptor'
    names = ['akt1', 'ampc', 'cp3a4', 'cxcr4', 'gcr', 'hivpr', 'hivrt', 'kif11',
             'KAT2A', 'MAPK1', 'MTORC1', 'VDR', 'PKM2', 'TP53', 'ALDH1', 'ESR1_ant', 'GBA', 'FEN1']
    for name in names:
        global path_name, path_name_lig, pre_protein_file, csv_file
        path_name = '%s/%s' % (path, name)
        protein_file = '%s/%s_protein.pdb' % (path_name, name)
        # write csv file
        csv_file = '%s/affini.csv' % path_name
        pd.DataFrame(
            ['name', 'Orientation_Score', 'Score(Heavy_Ligand_Atoms)', 'Affinity_Score',
                   'Buried_Protein_Hydrophobic_Term', 'Hydrophobic_Complementarity_Term', 'Polar_Component_Term',
                   'Number_of_Protein-Ligand_Hydrophobic_Contacts', 'Number_of_Protein-Ligand_H-bonds',
                   'Number_of_Protein-Ligand_Salt-bridges', 'Number_of_Metal-Ligand_Bonds',
                   'Number_of_Interfacial_Unsatisfied_Polar_Atoms', 'Number_of_Interfacial_Unsatisfied_Charged_Atoms',
                   'Buried_Carbons']).T.to_csv(csv_file, index=None, header=None)
        # prepare protein
        protein_temp_file = '%s/%s_temp.pdb' % (path_name, name)
        pre_protein_file = '%s/%s_p.pdb' % (path_name, name)
        if not os.path.exists(pre_protein_file):
            cmdline = 'module load amber &&'
            cmdline += 'reduce -Trim %s > %s &&' % (protein_file, protein_temp_file)  # 准备蛋白
            cmdline += 'cat %s | awk \'{if ($1!="ATOM" || $4!="ACE") print $0}\' > %s ' % \
                       (protein_temp_file, pre_protein_file)
            os.system(cmdline)
            if os.path.exists(protein_temp_file):
                os.remove(protein_temp_file)
        # get ligands
        path_name_lig = '%s/lig' % path_name
        ligands = os.listdir(path_name_lig)
        # multicore
        pool = Pool(24)
        pool.map(affini, ligands)
        pool.close()
        pool.join()

if __name__ == '__main__':
    cal_aff()
