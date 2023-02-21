# TB-IEC-Score: An accurate and efficient machine learning-based scoring function for virtual screening
## 1. To train a TB_IECS:
```
Usage:

python3 tb_iecs.py --protein_file [path/to/protein/file] --train_size [float number] --crystal_ligand_file [path/to/crystalized/ligand/file] --tmp_dir [temporary/directory/path] --dst_dir [directory/path/for/saving/files] --ligand_path [path/to/ligands/files]

Arguments:
--protein_file: The path of the protein PDB file.
--train_size: A float number used in the train-validation split.
--crystal_ligand_file: The path of the crystalized ligand file in MOL2 format.
--tmp_dir: The temporary directory for file saving.
--dst_dir: The directory for saving descriptors CSV file and model file.
--ligand_path: The path for ligands files.
```
## 2. scripts used for descriptors generation
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

## 3. Feature combination
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
## 4.Machine learning algorithms
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
