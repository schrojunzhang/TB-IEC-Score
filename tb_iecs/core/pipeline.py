"""
Main pipeline implementation for TB-IEC-Score
"""
import os
import subprocess
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from joblib import dump, load
from tb_iecs.core.model import XGBoostModel, get_model
from tb_iecs.descriptors.smina import process_ligand, smina_header


class TBIECPipeline:
    """
    Main pipeline for TB-IEC-Score that handles the full workflow from
    protein preparation to model training/prediction
    """
    
    def __init__(
        self,
        protein_file: str,
        crystal_ligand_file: str,
        dst_dir: str,
        smina_path: Optional[str] = None,
        nnscore_path: Optional[str] = None,
        mgltool_path: Optional[str] = None,
    ):
        """
        Initialize TB-IEC-Score pipeline.
        
        Args:
            protein_file: Path to protein PDB file
            crystal_ligand_file: Path to crystal ligand MOL2 file for binding site location
            dst_dir: Directory for result file saving
            smina_path: Path to smina installation (default: from environment variable)
            nnscore_path: Path to NNScore installation (default: from environment variable)
            mgltool_path: Path to MGLTools installation (default: from environment variable)
        """
        self.protein_file = protein_file
        self.crystal_ligand_file = crystal_ligand_file
        self.dst_dir = dst_dir
        
        # Create destination directory if it doesn't exist
        os.makedirs(dst_dir, exist_ok=True)
        
        # Get paths from environment if not provided
        self.smina_path = smina_path or os.environ.get('SMINA')
        self.nnscore_path = nnscore_path or os.environ.get('NNSCORE')
        self.mgltool_path = mgltool_path or os.environ.get('MGLTOOL')
        
        # Check required paths
        if not self.smina_path:
            raise ValueError("SMINA path not provided or set in environment")
        if not self.nnscore_path:
            raise ValueError("NNSCORE path not provided or set in environment")
        if not self.mgltool_path:
            raise ValueError("MGLTOOL path not provided or set in environment")
        
        # Prepare files
        self._prepare_files()
    
    def _prepare_files(self):
        """Prepare protein and ligand files for docking"""
        # Get file paths and names
        pro_path, pro_name = os.path.split(self.protein_file)
        cl_path, cl_name = os.path.split(self.crystal_ligand_file)
        
        # Define output file paths
        self.protein_pre = os.path.join(self.dst_dir, pro_name.replace('.pdb', '_smina_p.pdb'))
        self.protein_pred = os.path.join(self.dst_dir, pro_name.replace('.pdb', '_smina_p2.pdbqt'))
        self.crystal_file = os.path.join(self.dst_dir, cl_name.replace('.mol2', '.pdbqt'))
        
        # Convert crystal ligand to PDBQT if needed
        if not os.path.exists(self.crystal_file):
            cmd = (f'{self.mgltool_path}/bin/pythonsh '
                 f'{self.mgltool_path}/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py '
                 f'-l {self.crystal_ligand_file} -o {self.crystal_file} -A hydrogens')
            subprocess.run(cmd, shell=True, check=True)
        
        # Get binding pocket coordinates
        x = float(subprocess.check_output(
            f"cat {self.crystal_file} | awk '{{if ($1==\"ATOM\") print $(NF-6)}}' | "
            f"awk '{{x+=$1}} END {{print x/(NR)}}'", 
            shell=True).decode('utf-8').strip())
        
        y = float(subprocess.check_output(
            f"cat {self.crystal_file} | awk '{{if ($1==\"ATOM\") print $(NF-5)}}' | "
            f"awk '{{y+=$1}} END {{print y/(NR)}}'", 
            shell=True).decode('utf-8').strip())
        
        z = float(subprocess.check_output(
            f"cat {self.crystal_file} | awk '{{if ($1==\"ATOM\") print $(NF-4)}}' | "
            f"awk '{{z+=$1}} END {{print z/(NR)}}'", 
            shell=True).decode('utf-8').strip())
        
        self.pocket_center = (x, y, z)
        
        # Prepare protein file
        if not os.path.exists(self.protein_pred):
            cmd = (f'cat {self.protein_file} | sed \'/HETATM/\'d > {self.protein_pre} && '
                 f'{self.mgltool_path}/bin/pythonsh '
                 f'{self.mgltool_path}/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py '
                 f'-r {self.protein_pre} -o {self.protein_pred} -A hydrogens '
                 f'-U nphs_lps_waters_nonstdres && '
                 f'rm {self.protein_pre}')
            subprocess.run(cmd, shell=True, check=True)
        
        # Create CSV file for results
        self.smina_csv = os.path.join(self.dst_dir, 'smina.csv')
        pd.DataFrame(['name'] + smina_header).T.to_csv(
            self.smina_csv, index=None, header=None)
        
        # Create PDBQT directory for ligands
        self.pdbqt_dir = os.path.join(self.dst_dir, 'lig_pdbqt')
        os.makedirs(self.pdbqt_dir, exist_ok=True)
    
    def _generate_descriptors(self, ligand_path: str):
        """
        Generate descriptors for all ligands in the given path.
        
        Args:
            ligand_path: Path to ligand files directory
        """
        # Get all ligands
        ligands = os.listdir(ligand_path)
        
        # Calculate SMINA scores using multiprocessing
        with Pool() as pool:
            pool.map(
                lambda ligand: process_ligand(
                    ligand, 
                    ligand_path=ligand_path,
                    protein_pred=self.protein_pred,
                    pdbqt_dir=self.pdbqt_dir,
                    smina_path=self.smina_path,
                    pocket_center=self.pocket_center,
                    smina_csv=self.smina_csv
                ), 
                ligands
            )
        
        # Calculate NNScore descriptors
        cmd = f'python2 {os.path.join(os.path.dirname(__file__), "../../descriptors/nnscore.py")} {self.dst_dir} {ligand_path} {self.dst_dir}'
        subprocess.run(cmd, shell=True, check=True)
        
        # Merge descriptors
        df_smina = pd.read_csv(self.smina_csv)
        df_nn = pd.read_csv(os.path.join(self.dst_dir, 'nnscore.csv'))
        df = pd.merge(df_smina, df_nn, on='name', how='inner')
        
        return df
    
    def train(
        self, 
        ligand_path: str, 
        label_csv: str, 
        model_file: str,
        model_type: str = "xgboost",
        hyper_opt: bool = True
    ) -> Dict:
        """
        Train a TB-IEC-Score model.
        
        Args:
            ligand_path: Path to ligand files directory
            label_csv: Path to CSV file with ligand labels (name, label)
            model_file: Path to save the trained model
            model_type: Model type ('xgboost', 'svm', 'rf')
            hyper_opt: Whether to optimize hyperparameters
            
        Returns:
            Dictionary with training metrics
        """
        # Generate descriptors
        df = self._generate_descriptors(ligand_path)
        
        # Merge with labels
        df = pd.merge(df, pd.read_csv(label_csv), on='name', how='inner')
        
        # Prepare data for training
        x = df.loc[:, [col for col in df.columns if col not in ['name', 'label']]]
        y = df.loc[:, 'label']
        
        # Create and train model
        model = get_model(model_type=model_type, hyper_opt=hyper_opt)
        metrics, _ = model.train(x, y)
        
        # Save model
        dump((model.scaler, model.threshold, model.normalizer, model.clf), model_file)
        
        return metrics
    
    def predict(
        self,
        ligand_path: str,
        model_file: str
    ) -> pd.DataFrame:
        """
        Make predictions with a trained TB-IEC-Score model.
        
        Args:
            ligand_path: Path to ligand files directory
            model_file: Path to the trained model file
            
        Returns:
            DataFrame with prediction results
        """
        # Generate descriptors
        df = self._generate_descriptors(ligand_path)
        
        # Prepare data for prediction
        x = df.loc[:, [col for col in df.columns if col not in ['name', 'label']]]
        
        # Load model
        scaler, threshold, normalizer, clf = load(model_file)
        
        # Create model instance
        model = XGBoostModel(hyper_opt=False)
        model.scaler = scaler
        model.threshold = threshold
        model.normalizer = normalizer
        model.clf = clf
        
        # Make predictions
        pred, pred_proba = model.predict(x)
        
        # Add predictions to dataframe
        df['pred_y'] = pred
        df['pred_y_proba'] = pred_proba
        
        # Save results
        df.to_csv(os.path.join(self.dst_dir, 'result.csv'), index=None)
        
        return df 