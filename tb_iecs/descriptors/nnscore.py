"""
NNScore 2.0 energy terms extraction module
"""
import os
import subprocess
from typing import List, Optional

import pandas as pd

# NNScore 2.0 descriptor headers
nnscore_header = [
    "Hydrogen_bond_donors", "Hydrogen_bond_acceptors", "sp2_carbon_atoms",
    "aromatic_rings", "rotatable_bonds", "mrt_1a", "mrt_2a", "mrt_3a", "mrt_4a",
    "mrt_5a", "mrt_6a", "mrt_7a", "mrt_8a", "mrt_9a", "mrt_10a",
    "mrt_11a", "mrt_12a", "mrt_13a", "mrt_14a", "mrt_15a", "mrt_16a",
    "mrt_17a", "mrt_18a", "mrt_19a", "mrt_20a", "mrt_1b", "mrt_2b",
    "mrt_3b", "mrt_4b", "mrt_5b", "mrt_6b", "mrt_7b", "mrt_8b",
    "mrt_9b", "mrt_10b", "mrt_11b", "mrt_12b", "mrt_13b", "mrt_14b",
    "mrt_15b", "mrt_16b", "mrt_17b", "mrt_18b", "mrt_19b", "mrt_20b",
    "mrt_1c", "mrt_2c", "mrt_3c", "mrt_4c", "mrt_5c", "mrt_6c",
    "mrt_7c", "mrt_8c", "mrt_9c", "mrt_10c", "mrt_11c", "mrt_12c",
    "mrt_13c", "mrt_14c", "mrt_15c", "mrt_16c", "mrt_17c", "mrt_18c",
    "mrt_19c", "mrt_20c", "vina_total", "vina_gauss1", "vina_gauss2",
    "vina_repulsion", "vina_hydrophobic", "vina_hydrogen", "nn1_output", 
    "nn2_output", "nn3_output", "nn_output_average"
]


def run_nnscore(
    dst_dir: str,
    ligand_path: str,
    protein_file: str,
    pdbqt_dir: str,
    nnscore_path: Optional[str] = None
) -> None:
    """
    Run NNScore 2.0 to extract neural network-based descriptors for all ligands
    
    Args:
        dst_dir: Destination directory for output
        ligand_path: Path to ligand directory
        protein_file: Path to prepared protein file
        pdbqt_dir: Directory with converted PDBQT ligands
        nnscore_path: Path to NNScore 2.0 installation
    """
    # Get NNScore path from environment if not provided
    nnscore_path = nnscore_path or os.environ.get('NNSCORE')
    if not nnscore_path:
        raise ValueError("NNSCORE path not provided or set in environment")
    
    # Create output CSV file
    output_csv = os.path.join(dst_dir, 'nnscore.csv')
    
    # Write header to CSV
    pd.DataFrame([['name'] + nnscore_header]).to_csv(output_csv, index=False, header=False)
    
    # Get list of ligands
    ligands = os.listdir(ligand_path)
    
    # Process each ligand
    for ligand in ligands:
        # Skip non-SDF files
        if not ligand.endswith('.sdf'):
            continue
        
        # Get base name without extension
        ligand_base = os.path.splitext(ligand)[0]
        
        # Get PDBQT path
        ligand_pdbqt = os.path.join(pdbqt_dir, f"{ligand_base}.pdbqt")
        
        # Skip if PDBQT file doesn't exist
        if not os.path.exists(ligand_pdbqt):
            continue
        
        # Run NNScore 2.0
        cmd = (
            f"cd {nnscore_path} && "
            f"python2 {nnscore_path}/NNScore.py "
            f"-receptor {protein_file} -ligand {ligand_pdbqt} "
            f"-vina_executable {os.path.join(nnscore_path, 'vina')} "
        )
        
        try:
            # Run NNScore
            result = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL)
            
            # Parse output to get descriptors
            output = result.decode('utf-8').strip()
            
            # Extract descriptor values
            values = extract_nnscore_values(output)
            
            if values and len(values) == len(nnscore_header):
                # Create dataframe with ligand name and descriptor values
                df = pd.DataFrame([[ligand_base] + values])
                
                # Append to CSV file
                df.to_csv(output_csv, mode='a', header=False, index=False)
        except subprocess.CalledProcessError:
            print(f"Failed to run NNScore on {ligand}")
            continue


def extract_nnscore_values(output: str) -> List[float]:
    """
    Extract NNScore descriptor values from output text
    
    Args:
        output: NNScore output text
        
    Returns:
        List of descriptor values
    """
    lines = output.split('\n')
    values = []
    
    # Extract ligand properties
    for line in lines:
        if "Hydrogen-bond donors:" in line:
            values.append(float(line.split(':')[1].strip()))
        elif "Hydrogen-bond acceptors:" in line:
            values.append(float(line.split(':')[1].strip()))
        elif "sp2 carbon atoms:" in line:
            values.append(float(line.split(':')[1].strip()))
        elif "Aromatic rings:" in line:
            values.append(float(line.split(':')[1].strip()))
        elif "Rotatable bonds:" in line:
            values.append(float(line.split(':')[1].strip()))
    
    # Extract MRT values
    mrt_values = []
    for line in lines:
        if line.startswith("MRT"):
            parts = line.split()
            if len(parts) >= 3:
                mrt_values.append(float(parts[2]))
    
    # Add MRT values to results
    values.extend(mrt_values)
    
    # Extract Vina scores
    vina_values = []
    for line in lines:
        if line.startswith("Vina output:"):
            parts = line.split()
            if len(parts) >= 8:
                vina_values.extend([
                    float(parts[2]),  # total
                    float(parts[3]),  # gauss1
                    float(parts[4]),  # gauss2
                    float(parts[5]),  # repulsion
                    float(parts[6]),  # hydrophobic
                    float(parts[7])   # hydrogen
                ])
    
    # Add Vina values to results
    values.extend(vina_values)
    
    # Extract neural network outputs
    nn_values = []
    for line in lines:
        if line.startswith("NNScore 1:") or line.startswith("NNScore 2:") or line.startswith("NNScore 3:"):
            parts = line.split(':')
            if len(parts) >= 2:
                nn_values.append(float(parts[1].strip()))
    
    # Calculate average NN output if we have all three
    if len(nn_values) == 3:
        nn_values.append(sum(nn_values) / 3)
    
    # Add NN values to results
    values.extend(nn_values)
    
    return values


def main(dst_dir: str, ligand_path: str, result_dir: str):
    """
    Main function to run NNScore on all ligands
    
    Args:
        dst_dir: Directory with converted files
        ligand_path: Path to ligand directory
        result_dir: Directory for result output
    """
    # Get paths
    pdbqt_dir = os.path.join(dst_dir, 'lig_pdbqt')
    protein_file = [f for f in os.listdir(dst_dir) if f.endswith('_smina_p2.pdbqt')][0]
    protein_file = os.path.join(dst_dir, protein_file)
    
    # Run NNScore
    run_nnscore(result_dir, ligand_path, protein_file, pdbqt_dir)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python nnscore.py <dst_dir> <ligand_path> <result_dir>")
        sys.exit(1)
    
    dst_dir = sys.argv[1]
    ligand_path = sys.argv[2]
    result_dir = sys.argv[3]
    
    main(dst_dir, ligand_path, result_dir) 