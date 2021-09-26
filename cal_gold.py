#!usr/bin/env python
# -*- coding:utf-8 -*-
import os
import pandas as pd
import shutil
from multiprocessing import Pool

import numpy as np


# from ccdc.protein import Protein
# from ccdc.io import MoleculeWriter
# used for cal ASP Chemscore ChemPLP Goldscore implemented in Gold

def gold_score(ligand):
    lig_name = ligand.split('.')[0]
    to_ligand = '%s/%s' % (path_name_lig, ligand)
    x, y, z = xyz
    path_name_lig_score = '%s/%s_gold' % (path_name, lig_name)
    if not os.path.exists(path_name_lig_score):
        os.mkdir(path_name_lig_score)
    for s in ['goldscore', 'chemscore', 'plp', 'asp']:
        tem_sdf = '%s/%s.sdf' % (path_name_lig_score, lig_name)
        config = '''  GOLD CONFIGURATION FILE

  AUTOMATIC SETTINGS
autoscale = 1

  POPULATION
popsiz = auto
select_pressure = auto
n_islands = auto
maxops = auto
niche_siz = auto

  GENETIC OPERATORS
pt_crosswt = auto
allele_mutatewt = auto
migratewt = auto

  FLOOD FILL
radius = 12
origin = %s %s %s
do_cavity = 0
floodfill_atom_no = 0
cavity_file = 
floodfill_center = point

  DATA FILES
ligand_data_file %s 10
param_file = DEFAULT
set_ligand_atom_types = 1
set_protein_atom_types = 0
directory = .
tordist_file = DEFAULT
make_subdirs = 0
save_lone_pairs = 1
fit_points_file = fit_pts.mol2      
read_fitpts = 0

  FLAGS
internal_ligand_h_bonds = 0
flip_free_corners = 0
match_ring_templates = 0
flip_amide_bonds = 0
flip_planar_n = 1 flip_ring_NRR flip_ring_NHR
flip_pyramidal_n = 0
rotate_carboxylic_oh = flip
use_tordist = 1
postprocess_bonds = 1
rotatable_bond_override_file = DEFAULT
solvate_all = 1

  TERMINATION
early_termination = 1
n_top_solutions = 3
rms_tolerance = 1.5

  CONSTRAINTS
force_constraints = 0

  COVALENT BONDING
covalent = 0

  SAVE OPTIONS
save_score_in_file = 1 unweighted
save_protein_torsions = 1
per_atom_scores = 1
concatenated_output = %s    
clean_up_option delete_all_solutions
clean_up_option delete_redundant_log_files
clean_up_option delete_all_initialised_ligands
clean_up_option delete_empty_directories
clean_up_option delete_rank_file
clean_up_option delete_all_log_files
output_file_format = MACCS

  FITNESS FUNCTION SETTINGS
initial_virtual_pt_match_max = 3
relative_ligand_energy = 1
gold_fitfunc_path = %s
start_vdw_linear_cutoff = 6
score_param_file = DEFAULT

  RUN TYPE
run_flag = RESCORE retrieve

  PROTEIN DATA
protein_datafile = %s  
''' % (x, y, z, to_ligand, tem_sdf, s, pre_protein_file)
        # config file
        config_file = '%s/%s.conf' % (path_name_lig_score, lig_name)
        score_csv = '%s/%s_%s_score.csv' % (path_name_lig_score, lig_name, s)
        with open(config_file, 'w')as f:
            f.write(config)
        # commandline
        cmdline = 'module load ccdc &&'
        cmdline += 'cd %s &&' % path_name_lig_score
        cmdline += 'gold_auto %s ' % config_file
        os.system(cmdline)
        # convert structure
        cmdline = 'module load openeye &&'
        cmdline += 'convert.py %s %s' % (tem_sdf, score_csv)  # sdf to csv
        os.system(cmdline)


def get_result(lig_name, s, path_name_lig_score):
    scoring_funtion_dic = {
        'asp': 'ASP',
        'goldscore': 'GoldScore',
        'plp': 'PLP',
        'chemscore': 'ChemScore'
    }
    scoring_function = scoring_funtion_dic[s]
    # score file
    score_csv = '%s/%s_%s_score.csv' % (path_name_lig_score, lig_name, s)
    if os.path.exists(score_csv):
        # read df
        df = pd.read_csv(score_csv, encoding='utf-8')

        def pro_data(df, score_function):
            # get data
            file_score = df.loc[0, 'Gold.{}.Protein.Score.Contributions'.format(score_function)]
            # # get active residue
            # active_res = df.loc[0, 'Gold.Protein.ActiveResidues'].split()
            # # remove duplicates
            # active_res = set([i for i in active_res if i != '|'])
            active_res = atom2res.values()
            # get list
            pro_features = file_score.split('|')
            # get interaction types
            pro_columns = pro_features[0].split()[1:]
            data_dic = {}
            # get interactions of residues
            for i in range(1, len(pro_features)):
                tmp_data = pro_features[i].split()
                res_name = atom2res[tmp_data[0]]
                if data_dic.get(res_name, None):
                    data_dic[res_name].append([float(j) for j in tmp_data[1:]])
                else:
                    data_dic[res_name] = [[float(j) for j in tmp_data[1:]]]
            # add interactions of each residue
            pro_columns_new = []
            features_lis = []
            all_columns_new = []
            for res, feature in data_dic.items():
                pro_columns_new.extend(['{}_{}'.format(res, i) for i in pro_columns])
                tmp_data = np.array(feature)
                features = list(np.sum(tmp_data, axis=0))
                features_lis.extend(features)
            # get columns
            for ac_res in active_res:
                all_columns_new.extend(['{}_{}'.format(ac_res, i) for i in pro_columns])
            # rename columns to 0
            new_dic = {}
            for i in all_columns_new:
                new_dic[i] = 0
            # get values to corresponding columns
            for i in range(len(pro_columns_new)):
                new_dic[pro_columns_new[i]] = features_lis[i]
            # get columns and data
            pro_data = new_dic.items()
            pro_columns, pro_features = [*zip(*pro_data)]
            return pro_features, pro_columns

        def lig_data(df, score_function):
            file_score = df.loc[0, 'Gold.{}.Ligand.Score.Contributions'.format(score_function)]
            file_score = file_score.split('|')
            columns = file_score[0].split()[1:]
            features = []
            file_columns = ['lig_{}'.format(j) for j in columns]
            for i in range(1, len(file_score)):
                features.append([float(j) for j in file_score[i].split()[1:]])
            features = np.array(features)
            features = list(np.sum(features, axis=0))
            return features, file_columns

        # get data
        lig_features, lig_columns = lig_data(df, score_function=scoring_function)
        try:
            pro_features, pro_columns = pro_data(df, score_function=scoring_function)
        except:
            print(lig_name)
            pro_features, pro_columns = [], []
        # merge data and columns of proteins and ligands
        lig_features.extend(pro_features)
        lig_columns.extend(pro_columns)
        lig_features.insert(0, lig_name)
        lig_columns.insert(0, 'name')
        # csv file
        csv_file = '%s/%s_IFP.csv' % (path_name, s)
        # # output to csv
        # if os.path.exists(csv_file):
        #     pd.DataFrame(lig_features, index=lig_columns).T.to_csv(csv_file, index=False, header=False, mode='a')
        # else:
        #     pd.DataFrame(lig_features, index=lig_columns).T.to_csv(csv_file, index=False)


def cal_gold():
    global path
    path = r'/home/xujun/Project_2/2_descriptor'
    names = ['akt1', 'ampc', 'cp3a4', 'cxcr4', 'gcr', 'hivpr', 'hivrt', 'kif11',
             'KAT2A', 'MAPK1', 'MTORC1', 'VDR', 'PKM2', 'TP53', 'ALDH1', 'ESR1_ant', 'GBA', 'FEN1']
    # , 'gcr', 'hivpr', 'hivrt', 'kif11' 'akt1',
    names = ['ampc', 'cp3a4', 'cxcr4']
    for name in names:
        global path_name, path_name_lig, pre_protein_file, xyz, atom2res
        path_name = '%s/%s' % (path, name)
        protein_file = '%s/%s_protein.pdb' % (path_name, name)
        crystal_file = '%s/%s_crystal_ligand.mol2' % (path_name, name)
        # prepare protein
        pre_protein_file = '%s/%s_g.pdb' % (path_name, name)
        if not os.path.exists(pre_protein_file):
            pass
            # mol = Protein.from_file('%s' % protein_file)
            # mol.remove_all_waters()
            # mol.remove_unknown_atoms()
            # mol.add_hydrogens()
            # with MoleculeWriter('%s' % pre_protein_file) as protein_writer:
            #     protein_writer.write(mol)
        # def atom2res():
        with open(pre_protein_file, 'r') as f:
            content = f.readlines()
        atom2res = {}
        for i in range(len(content)):
            if content[i].startswith('ATOM') or content[i].startswith('TER'):
                data = content[i].split()
                atom2res[data[1]] = '{}_{}'.format(data[5], data[3])
        # get pocket
        x = os.popen(
            "cat %s | sed -n '/@<TRIPOS>ATOM/,/@<TRIPOS>BOND/'p | awk '{print $3}' | awk '{x+=$1} END {print x/(NR-2)}'" % crystal_file).read()
        y = os.popen(
            "cat %s | sed -n '/@<TRIPOS>ATOM/,/@<TRIPOS>BOND/'p | awk '{print $4}' | awk '{y+=$1} END {print y/(NR-2)}'" % crystal_file).read()
        z = os.popen(
            "cat %s | sed -n '/@<TRIPOS>ATOM/,/@<TRIPOS>BOND/'p | awk '{print $5}' | awk '{z+=$1} END {print z/(NR-2)}'" % crystal_file).read()
        xyz = [x.strip(), y.strip(), z.strip()]
        # get ligand name
        path_name_lig = '%s/lig' % path_name
        ligands = os.listdir(path_name_lig)

        # collect result
        def fast_result(ligand):
            lig_name = ligand.split('.')[0]
            path_name_lig_score = '%s/%s_gold' % (path_name, lig_name)
            for s in ['goldscore', 'chemscore', 'plp', 'asp']:
                get_result(lig_name, s, path_name_lig_score)
            # remove file
            shutil.rmtree(path_name_lig_score)

        # multicore
        pool = Pool(28)
        pool.map(gold_score, ligands)
        pool.map(fast_result, ligands)
        pool.close()
        pool.join()


if __name__ == '__main__':
    cal_gold()
