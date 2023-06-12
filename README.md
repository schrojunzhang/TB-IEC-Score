# TB-IEC-Score: An accurate and efficient machine learning-based scoring function for virtual screening
## Installation
### Install smina:
```
download smina: https://zenodo.org/record/8025952
mkdir /root/smina
mv smina /root/smina/smina
mv total.score /root/smina/total.score
cd /root/smina
chmod +x ./smina
# smina_path = '/root/smina'
```
### install nnscore 2.0
```
download nnscore: https://zenodo.org/record/8025934
unzip -q NNscore.zip
mv NNscore /root/NNscore
# nnscore_path = '/root/NNscore'
```
### install mgltool
```
download mgltool: https://ccsb.scripps.edu/mgltools/download/491/
tar -zxvf mgltools_x86_64Linux2_1.5.7p1.tar.gz
cd mgltools_x86_64Linux2_1.5.7/
chmod +x install.sh
./install.sh
# mgltool_path = '/opt/mgltools/1.5.7'
```
### install python
```
sudo apt install python3
sudo apt install python2
pip3 install pandas
pip3 install numpy
pip3 install scikit-learn
pip3 install hyperopt
```
## To train/use a TB_IECS:
```
Usage:
# set the environment variables
export SMINA=/root/smina
export NNSCORE=/root/NNscore
export MGLTOOL=/opt/mgltools/1.5.7
# run the script
python3 tb_iecs.py --protein_file [path/to/protein/file] --label_csv [path/to/label_csv/file] --mode ['train', 'test'] --crystal_ligand_file [path/to/crystalized/ligand/file] --dst_dir [~/path/for/result_file_save] --model_file [~/path/for/model_pkl_save] --ligand_path [path/to/ligands/files]

Arguments:
--protein_file: The path of the protein PDB file.
--label_csv: The path of the csv file recoding the ligand's class [1:active or 0:inactive]. The csv file should contain two columns: 'name' and 'label'. 'name' represents the ligand's name, and 'label' represents the ligand's bio-activity.
--mode: choice from ['train', 'test']. If 'train', the model will be trained and saved. If 'test', the model will be loaded and used for prediction.
--crystal_ligand_file: The path of the crystalized ligand file in MOL2 format. Used for binding site locating.
--dst_dir: The directory for result file saving.
--model_file: The directory for saving/loading model file.
--ligand_path: The path for ligands files. multiple ligands files in SDF format are supported.

Training Example:

python3 tb_iecs.py --protein_file xx/1a30.pdb --label_csv xx/train_lig.csv --mode train --crystal_ligand_file xx/1a30.mol2 --dst_dir xx/result --model_file xx/tb_iec.pkl --ligand_path xx/training_ligands

Result: xx/tb_iec.pkl, a model pkl that can be used for prediction

Evaluation Example:

python3 tb_iecs.py --protein_file xx/1a30.pdb --label_csv xx/test_lig.csv --mode test --crystal_ligand_file xx/1a30.mol2 --dst_dir xx/result --model_file xx/tb_iec.pkl --ligand_path xx/test_ligands

Result: xx/result/result.csv, a csv file that records the predicted labels and predicted probabilities of each ligand
```
# other scripts related to experiments in the paper
## 1. scripts used for descriptors generation
```
Descriptions

cal_aff.py: defined the function for generating energy terms from Affiscore
cal_autodock.py: defined the function for generating energy terms from Autodock
cal_dsx.py: defined the function for generating energy terms from DSX
cal_galaxy.py: defined the function for generating energy terms from GalaxyDock BP2 Score
cal_gold.py: defined the functions for generating energy terms from ASP, Chemscore, ChemPLP and Goldscore
cal_smina.py: defined the function for generating energy terms from Smina
cal_nn.py: defined the function for generating energy terms from NNscore 2.0
cal_smo.py: defined the function for generating energy terms from SMoG2016
cal_vina.py: defined the function for generating energy terms from AutoDock Vina
cal_xp.py: defined the functions for generating energy terms from Glide sp and Glide xp
cal_xp.py: defined the function for generating energy terms from X-score
```
```
Usage:

1. download the datasets used in this study (http://dude.docking.org/ and http://drugdesign.unistra.fr/LIT-PCBA); 
2. install the corresponding scoring function you'd like to use;
(e.g, if you want to generate energy terms of SMoG2016, you need to install it and python cal_smo.py)
2. repalce the **name**, **path**, **csv_file** with your own values; 
(Here, "name" represents the targets, "path" denotes the directory where the dataset is stored, and "csv_file" represents the file path where the generated descriptors will be recorded in CSV format.)
3. python3 cal_xx.py
```

## 2. Feature combination
```
Descriptions

base_data.py: defined the dataset object for feature combination and importance calculation
cal_energy_importance.py: used for calculate the energy terms' importance score (Tree-based importance score)
```
```
Usage:

base_data.py  
1. for each target, generate all the energy terms using cal_xx.py and concat them to one csv
2. repalce the **__file_path**, **__score_path** with your own values.
(Here, "__file_path" represents the directory where the concated energy terms are saved, "__score_path" denotes the directory where the performance of models is stored.)

cal_energy_importance.py:
1. repalce the **names**, **dst_csv** with your own values; 
(Here, "names" represents the targets, "dst_csv" denotes the directory where the importance scores of various energy terms are stored.)
2. python cal_energy_importance.py

```
## 3. Machine learning algorithms
```
Descriptions

svm_model.py: This file is used for constructing SVM models and exploring the best feature combinations.
algo_compare.py: This file defines functions to construct SVM, RF, and XGBoost models, and is used for exploring the best machine learning algorithms.
```
```
Usage:

svm_model.py: 
1. modify the target;
2. python svm_model.py

algo_compare.py: 
1. modify **name**, **path_src**, ***path_dst*;
(Here, "names" represents the targets, "path_src" denotes the directory where the best feature combination are stored and "path_dst" represents the performance of various ML algorithms.)
2. python algo_compare.py
```
