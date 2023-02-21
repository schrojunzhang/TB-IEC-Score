#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   tb_iecs.py
@Time    :   2023/02/21 11:16:38
@Author  :   Xujun Zhang
@Version :   1.0
@Contact :   
@License :   
@Desc    :   None
'''

# here put the import lib
import os
import argparse
from joblib import dump
import pandas as pd
from multiprocessing import Pool
from algo_compare import xgb_model
from cal_smina import smina_score, smina_header 

# args
argparser = argparse.ArgumentParser()
argparser.add_argument('--protein_file', type=str, default='~/path/to/protein.pdb')
argparser.add_argument('--train_size', type=float, default=0.8)
argparser.add_argument('--crystal_ligand_file', type=str, default='~/path/to/crystal_ligand.mol2')
argparser.add_argument('--tmp_dir', type=str, default='~/path/for/tmp_file_save')
argparser.add_argument('--dst_dir', type=str, default='~/path/for/descriptors and model save')
argparser.add_argument('--ligand_path', type=str, default='~/path/to/ligand_dir')
args = argparser.parse_args()
# 
protein_file = args.protein_file
crystal_ligand_file = args.crystal_ligand_file
tmp_dir = args.tmp_dir
dst_dir = args.dst_dir
ligand_path = args.ligand_path
# init
pro_path, pro_name = os.path.split(protein_file)
cl_path, cl_name = os.path.split(crystal_ligand_file)
protein_pre = '%s/%s' % (tmp_dir, pro_name.replace('.pdb', '_smina_p.pdb'))
protein_pred = '%s/%s' % (tmp_dir, pro_name.replace('.pdb', '_smina_p2.pdbqt'))
crystal_file = '%s/%s' % (tmp_dir, cl_name.replace('.mol2', '.pdbqt'))
# move file and convert struct
if not os.path.exists(crystal_file):
    cmdline = 'module purge &&'
    cmdline += 'module load autodock &&'
    cmdline += 'prepare_ligand4.py -l %s -o %s -A hydrogens ' % (crystal_ligand_file, crystal_file)
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
csv_file = '%s/smina.csv' % dst_dir
pd.DataFrame(
    ['name'] + smina_header).T.to_csv(
    csv_file, index=None, header=None)
# get ligands
ligands = os.listdir(ligand_path)
# create pdbqt file folder
path_name_pdbqt = '%s/lig_pdbqt' % tmp_dir
os.makedirs(path_name_pdbqt, exist_ok=True)
# cal smina
pool = Pool()
pool.map(smina_score, ligands)
pool.close()
pool.join()
# cal nn
cmd = 'module load anaconda2 &&'
cmd += f'python2 ./cal_nn.py {tmp_dir} {ligand_path} {dst_dir}'
os.system(cmd)
# merge csv
df_smina = pd.read_csv(csv_file)
df_nn = pd.read_csv('%s/nnscore.csv' % dst_dir)
df = pd.merge(df_smina, df_nn, on='name', how='inner')
df['label'] = 'user defined'
# construct model
model = xgb_model(df, train_size=args.train_size, over_sampling=False, hyper_opt=True, return_model=True)
# save model
dump(model, f'{dst_dir}/tb_iecs.pkl')
