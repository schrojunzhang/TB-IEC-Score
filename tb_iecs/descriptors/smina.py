"""
SMINA energy terms extraction module
"""
import os
import subprocess
import pandas as pd
from typing import List, Tuple, Union, Dict, Optional


# SMINA energy term headers
smina_header = [
    "gauss1", "gauss2", "repulsion", "hydrophobic", 
    "non_dir_h_bond", "non_rot_hydrophobic", "rot_bond", 
    "catpi", "pi_pi", "non_dir_anti_h_bond", "non_dir_donor_h_bond", 
    "donor_donor", "acceptor_acceptor", "vdw", "electrostatic"
]


def process_ligand(
    ligand: str,
    ligand_path: str,
    protein_pred: str,
    pdbqt_dir: str,
    smina_path: str,
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
        pocket_center: Binding pocket center coordinates (x, y, z)
        smina_csv: Path to save SMINA energy terms CSV
    """
    # Get ligand file paths
    ligand_file = os.path.join(ligand_path, ligand)
    
    # Get base name without extension
    ligand_base = os.path.splitext(ligand)[0]
    
    # Skip if not SDF file
    if not ligand_file.endswith(".sdf"):
        return
    
    # Define output PDBQT path
    ligand_pdbqt = os.path.join(pdbqt_dir, f"{ligand_base}.pdbqt")
    
    # Convert ligand to PDBQT if needed
    if not os.path.exists(ligand_pdbqt):
        cmd = (f"{smina_path}/smina -i {ligand_file} -o {ligand_pdbqt}")
        try:
            subprocess.run(cmd, shell=True, check=True, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            print(f"Failed to convert {ligand} to PDBQT.")
            return
    
    # Extract pocket center coordinates
    x, y, z = pocket_center
    
    # Run SMINA to get energy terms
    cmd = (
        f"{smina_path}/smina -r {protein_pred} -l {ligand_pdbqt} "
        f"--scoring vinardo --custom_scoring {smina_path}/total.score "
        f"--center_x {x} --center_y {y} --center_z {z} --size_x 20 --size_y 20 --size_z 20 "
        f"--num_modes 1 -o /dev/null --score_only"
    )
    
    try:
        # Run SMINA scoring
        result = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL)
        
        # Parse output to get energy terms
        lines = result.decode('utf-8').strip().split('\n')
        
        # Find the energy terms line
        for line in lines:
            if line.startswith("Affinity:"):
                parts = line.split('|')
                if len(parts) > 1:
                    energy_parts = parts[1].strip().split()
                    if len(energy_parts) == len(smina_header):
                        # Create dataframe with ligand name and energy terms
                        df = pd.DataFrame([[ligand_base] + energy_parts])
                        
                        # Append to CSV file
                        df.to_csv(smina_csv, mode='a', header=False, index=False)
                        break
    except subprocess.CalledProcessError:
        print(f"Failed to score {ligand} with SMINA.")
        return 