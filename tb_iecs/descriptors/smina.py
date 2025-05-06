"""
SMINA energy terms extraction module
"""
import os
import subprocess
import pandas as pd
from typing import List, Tuple, Union, Dict, Optional


# SMINA energy term headers
smina_header = ['gauss(o=0,_w=0.3,_c=8)', 'gauss(o=0.5,_w=0.3,_c=8)', 'gauss(o=1,_w=0.3,_c=8)',
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
             'num_hydrophobic_atoms', 'ligand_length', 'num_ligands']


def process_ligand(
    ligand: str,
    ligand_path: str,
    protein_pred: str,
    pdbqt_dir: str,
    smina_path: str,
    mgl_tool_path: str,
    pocket_center: Tuple[float, float, float],
    smina_csv: str
) -> None:
    """
    Process a ligand to extract SMINA energy terms
    
    Args:
        ligand: Ligand filename
        ligand_path: Path to ligand directory
        protein_pred: Path to prepared protein PDBQT file
        pdbqt_dir: Directory to store converted PDBQT ligands
        smina_path: Path to SMINA installation
        mgl_tool_path: Path to MGLTools installation
        pocket_center: Binding pocket center coordinates (x, y, z)
        smina_csv: Path to save SMINA energy terms CSV
    """
    # Get ligand file paths
    ligand_file = os.path.join(ligand_path, ligand)
    
    # Get base name without extension
    ligand_base = os.path.splitext(ligand)[0]
    
    # Skip if not mol2 file
    if not ligand_file.endswith(".mol2"):
        return
    
    # Define output PDBQT path
    ligand_pdbqt = os.path.join(pdbqt_dir, f"{ligand_base}.pdbqt")
    
    # Convert ligand to PDBQT if needed
    if not os.path.exists(ligand_pdbqt):
        cmd = f'cd {ligand_path} && '
        cmd += (f"{mgl_tool_path}/bin/pythonsh {mgl_tool_path}/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py -l {ligand_file} -o {ligand_pdbqt} -A hydrogens")
        try:
            subprocess.run(cmd, shell=True, check=True, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            print(f"Failed to convert {ligand} to PDBQT based on the command: {cmd}")
            return
    
    # Extract pocket center coordinates
    x, y, z = pocket_center
    
    # Run SMINA to get energy terms
    cmd = (
        f"{smina_path}/smina --receptor {protein_pred} --ligand {ligand_pdbqt} "
        f"--custom_scoring {smina_path}/total.score "
        f"--center_x {x} --center_y {y} --center_z {z} --size_x 18.75 --size_y 18.75 --size_z 18.75 "
        f"--cpu 1 -o /dev/null --score_only"
    )
    
    try:
        # Run SMINA scoring
        result = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL)
        
        # Parse output to get energy terms
        lines = result.decode('utf-8').strip().split('\n')
        '''
            result = [lig_name] + [0 for i in range(81)]
    for i in range(len(con)):
        if con[i].startswith('Term values, before weighting:'):
            result = [lig_name] + [x.strip() for x in con[i + 1].lstrip('##  ').split(' ')]
        '''
        # Find the energy terms line
        result = [ligand_base] + [0 for i in range(81)]
        for i, line in enumerate(lines):
            if line.startswith("Term values, before weighting:"):
                result = [ligand_base] + [x.strip() for x in lines[i + 1].lstrip('##  ').split(' ')]
                break
        # Create dataframe with ligand name and energy terms
        df = pd.DataFrame([result])
        # Append to CSV file
        df.to_csv(smina_csv, mode='a', header=False, index=False)
    except subprocess.CalledProcessError:
        print(f"Failed to score {ligand} with SMINA based on the command: {cmd}")
        return 