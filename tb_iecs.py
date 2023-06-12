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
from joblib import dump, load
import pandas as pd
from multiprocessing import Pool
from algo_compare import xgb_model_for_tbiecs
from cal_smina import smina_score, smina_header 
mgl_tool_path = os.environ.get('MGLTOOL') or '/opt/mgltools/1.5.7' 

# args
argparser = argparse.ArgumentParser()
argparser.add_argument('--protein_file', type=str, default='~/path/to/protein.pdb')
argparser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
argparser.add_argument('--label_csv', type=str, default='~/path/for/label_csv')
argparser.add_argument('--crystal_ligand_file', type=str, default='~/path/to/crystal_ligand.mol2')
argparser.add_argument('--dst_dir', type=str, default='~/path/for/result_file_save')
argparser.add_argument('--model_file', type=str, default='~/path/for/model_pkl_save')
argparser.add_argument('--ligand_path', type=str, default='~/path/to/ligand_dir')
args = argparser.parse_args()
# 
protein_file = args.protein_file
crystal_ligand_file = args.crystal_ligand_file
dst_dir = args.dst_dir
model_file = args.model_file
ligand_path = args.ligand_path
os.makedirs(dst_dir, exist_ok=True)
# init
pro_path, pro_name = os.path.split(protein_file)
cl_path, cl_name = os.path.split(crystal_ligand_file)
protein_pre = '%s/%s' % (dst_dir, pro_name.replace('.pdb', '_smina_p.pdb'))
protein_pred = '%s/%s' % (dst_dir, pro_name.replace('.pdb', '_smina_p2.pdbqt'))
crystal_file = '%s/%s' % (dst_dir, cl_name.replace('.mol2', '.pdbqt'))
# move file and convert struct
if not os.path.exists(crystal_file):
    # cmdline = 'module purge &&'
    # cmdline += 'module load autodock &&'
    cmdline = '%s/bin/pythonsh %s/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py -l %s -o %s -A hydrogens ' % (mgl_tool_path, mgl_tool_path, crystal_ligand_file, crystal_file)
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
    # cmdline += 'module purge &&'
    # cmdline += 'module load autodock &&'
    cmdline += '%s/bin/pythonsh %s/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py -r %s -o %s -A hydrogens -U nphs_lps_waters_nonstdres &&' % (
        mgl_tool_path, mgl_tool_path, protein_pre, protein_pred)
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
path_name_pdbqt = '%s/lig_pdbqt' % dst_dir
os.makedirs(path_name_pdbqt, exist_ok=True)
# cal smina
pool = Pool()
pool.map(smina_score, ligands)
pool.close()
pool.join()
# cal nn
cmd = f'python2 ./cal_nn.py {dst_dir} {ligand_path} {dst_dir}'
os.system(cmd)
# merge csv
df_smina = pd.read_csv(csv_file)
df_nn = pd.read_csv('%s/nnscore.csv' % dst_dir)
df = pd.merge(df_smina, df_nn, on='name', how='inner')
# construct model
if args.mode == 'train':
    df = pd.merge(df, pd.read_csv(args.label_csv), on='name', how='inner')
    x = df.loc[:, [i for i in df.columns if i not in ['name', 'label']]]
    y = df.loc[:, 'label']
    model = xgb_model_for_tbiecs(x=x, y=y, hyper_opt=True)
    # save model
    dump(model, model_file)
else:
    x = df.loc[:, [i for i in df.columns if i not in ['name', 'label']]]
    scaler, threshold, normalizer, clf = load(model_file)
    pred_y, pred_y_proba = xgb_model_for_tbiecs(x=x, hyper_opt=False, 
                                                model=clf,
                                                scaler=scaler,
                                                threshold=threshold,
                                                normalizer=normalizer,
                                                )
    df['pred_y'] = pred_y
    df['pred_y_proba'] = pred_y_proba
    df.to_csv('%s/result.csv' % dst_dir, index=None)