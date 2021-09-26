#! usr/bin/env python3
# -*- coding:utf-8 -*-
# used for cal xscore
import os
import pandas as pd
from multiprocessing import Pool


def cal_xscore(ligand):
    ligand_file = '%s/%s' % (path_local_lig, ligand)
    lig_name = ligand.split('.')[0]
    table_file = '%s/%s.table' % (path_local, lig_name)
    log_file = '%s/%s.log' % (path_local, lig_name)
    mdb_file = '%s/xscore.mdb ' % path_local
    para = '''
#######################################################################
#                            XTOOL/SCORE                             # 
######################################################################
###
FUNCTION	SCORE
###
### set up input and output files ------------------------------------
###
#
RECEPTOR_PDB_FILE    %s
#prepare protein
FixPDB           
#非必需
#REFERENCE_MOL2_FILE  ./1ppc_ligand.mol2
#COFACTOR_MOL2_FILE  none               
LIGAND_MOL2_FILE     %s
#prepare ligand
FixMol2
#
OUTPUT_TABLE_FILE    %s
OUTPUT_LOG_FILE      %s
###
### how many top hits to extract from the LIGAND_MOL2_FILE?
###
NUMBER_OF_HITS       1 
HITS_DIRECTORY       %s
###
### want to include atomic binding scores in the resulting Mol2 files?
###
SHOW_ATOM_BIND_SCORE	YES		[YES/NO]
###
### set up scoring functions -----------------------------------------
###
APPLY_HPSCORE         YES             	[YES/NO]
    HPSCORE_CVDW  0.004 
    HPSCORE_CHB   0.053
    HPSCORE_CHP   0.011
    HPSCORE_CRT  -0.061
    HPSCORE_C0    3.448
APPLY_HMSCORE         YES             	[YES/NO]
    HMSCORE_CVDW  0.004
    HMSCORE_CHB   0.094
    HMSCORE_CHM   0.394
    HMSCORE_CRT  -0.099
    HMSCORE_C0    3.585
APPLY_HSSCORE         YES 	  	[YES/NO]
    HSSCORE_CVDW  0.004
    HSSCORE_CHB   0.069
    HSSCORE_CHS   0.004
    HSSCORE_CRT  -0.092
    HSSCORE_C0    3.349
###
### set up chemical rules for pre-screening ligand molecules ---------
###（类药性五规则）
APPLY_CHEMICAL_RULES    NO            [YES/NO]	
    MAXIMAL_MOLECULAR_WEIGHT      600.0
    MINIMAL_MOLECULAR_WEIGHT      200.0
    MAXIMAL_LOGP                  6.00
    MINIMAL_LOGP                  1.00
    MAXIMAL_HB_ATOM               8 
    MINIMAL_HB_ATOM               2 
###

###
#xscore input_parameter_file
                    ''' % (protein_file, ligand_file, table_file, log_file, mdb_file)
    # write config file
    para_file = '%s/%s.input' % (path_local, lig_name)
    with open(para_file, 'w') as f:
        f.write(para)
    cmdline = 'cd /home/xujun/Soft/Score_Function/xscorelinux/bin &&'
    cmdline += './xscore %s ' % para_file
    os.system(cmdline)
    # get descriptors
    energy = [lig_name] + [0 for x in range(7)]
    with open(log_file, 'r') as f:
        con = f.readlines()
    for i in range(len(con)):
        if 'Total' in con[i]:
            energy = [x.strip('\n') for x in con[i].split(' ') if x != '']

    energy[0] = lig_name
    pd.DataFrame(energy).T.to_csv(csv_file, mode='a', header=None, index=None)
    # remove files
    os.remove(table_file)
    os.remove(log_file)
    os.remove(para_file)


def cal_all_xscore():
    global path
    path = r'/home/xujun/Project_2/2_descriptor'
    names = ['akt1', 'ampc', 'cp3a4', 'cxcr4', 'gcr', 'hivpr', 'hivrt', 'kif11',
             'KAT2A', 'MAPK1', 'MTORC1', 'VDR', 'PKM2', 'TP53', 'ALDH1', 'ESR1_ant', 'GBA', 'FEN1']
    for name in names:
        global path_local, path_local_lig, protein_file, csv_file
        path_local = '%s/%s' % (path, name)
        path_local_lig = '%s/lig' % path_local
        protein_file = '%s/%s_protein.pdb' % (path_local, name)
        ligands = os.listdir(path_local_lig)
        csv_file = '%s/xscore.csv' % path_local
        pd.DataFrame(['name', 'VDW', 'HB', 'HP', 'HM', 'HS', 'RT', 'XSCORE']).T.to_csv(csv_file, index=None,
                                                                                       header=None)
        pool = Pool(28)
        pool.map(cal_xscore, ligands)
        pool.close()
        pool.join()


if __name__ == '__main__':
    cal_all_xscore()
