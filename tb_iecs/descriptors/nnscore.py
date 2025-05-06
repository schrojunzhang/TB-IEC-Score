"""
NNScore 2.0 energy terms extraction module
"""
import os
import subprocess
from typing import List, Optional

import pandas as pd

# NNScore 2.0 descriptor headers
# define atom types
vina_output_list = ['vina_affinity', 'vina_gauss_1', 'vina_gauss_2', 'vina_repulsion', 'vina_hydrophobic',
                    'vina_hydrogen']
ligand_receptor_atom_type_pairs_less_than_two_half_list = ['A_MN', 'OA_SA', 'HD_N', 'N_ZN', 'A_MG', 'HD_NA', 'A_CL',
                                                            'MG_OA', 'FE_HD', 'A_OA', 'NA_ZN', 'A_N', 'C_OA', 'F_HD',
                                                            'C_HD', 'NA_SA', 'A_ZN', 'C_NA', 'N_N', 'MN_N', 'F_N',
                                                            'FE_OA', 'HD_I', 'BR_C', 'MG_NA', 'C_ZN', 'CL_MG',
                                                            'BR_OA',
                                                            'A_FE', 'CL_OA', 'CL_N', 'NA_OA', 'F_ZN', 'HD_P',
                                                            'CL_ZN',
                                                            'C_C', 'C_CL', 'FE_N', 'HD_S', 'HD_MG', 'C_F', 'A_NA',
                                                            'BR_HD', 'HD_OA', 'HD_MN', 'A_SA', 'A_F', 'HD_SA', 'A_C',
                                                            'A_A', 'F_SA', 'C_N', 'HD_ZN', 'OA_OA', 'N_SA', 'CL_FE',
                                                            'C_MN', 'CL_HD', 'OA_ZN', 'MN_OA', 'C_MG', 'F_OA',
                                                            'CD_OA',
                                                            'S_ZN', 'N_OA', 'C_SA', 'N_NA', 'A_HD', 'HD_HD', 'SA_ZN']
ligand_receptor_atom_type_pairs_less_than_four_list = ['I_N', 'OA_SA', 'FE_NA', 'HD_NA', 'A_CL', 'MG_SA', 'A_CU',
                                                        'P_SA', 'C_NA', 'MN_NA', 'F_N', 'HD_N', 'HD_I', 'CL_MG',
                                                        'HD_S',
                                                        'CL_MN', 'F_OA', 'HD_OA', 'F_HD', 'A_SA', 'A_BR', 'BR_HD',
                                                        'SA_SA', 'A_MN', 'N_ZN', 'A_MG', 'I_OA', 'C_C', 'N_S', 'N_N',
                                                        'FE_N', 'NA_SA', 'BR_N', 'MN_N', 'A_P', 'BR_C', 'A_FE',
                                                        'MN_P',
                                                        'CL_OA', 'CU_HD', 'MN_S', 'A_S', 'FE_OA', 'NA_ZN', 'P_ZN',
                                                        'A_F',
                                                        'A_C', 'A_A', 'A_N', 'HD_MN', 'A_I', 'N_SA', 'C_OA', 'MG_P',
                                                        'BR_SA', 'CU_N', 'MN_OA', 'MG_N', 'HD_HD', 'C_FE', 'CL_NA',
                                                        'MG_OA', 'A_OA', 'CL_ZN', 'BR_OA', 'HD_ZN', 'HD_P', 'OA_P',
                                                        'OA_S', 'N_P', 'A_NA', 'CL_FE', 'HD_SA', 'C_MN', 'CL_HD',
                                                        'C_MG',
                                                        'FE_HD', 'MG_S', 'NA_S', 'NA_P', 'FE_SA', 'P_S', 'C_HD',
                                                        'A_ZN',
                                                        'CL_P', 'S_SA', 'CL_S', 'OA_ZN', 'N_NA', 'MN_SA', 'CL_N',
                                                        'NA_OA', 'C_ZN', 'C_CD', 'HD_MG', 'C_F', 'C_I', 'C_CL',
                                                        'C_N',
                                                        'C_P', 'C_S', 'A_HD', 'F_SA', 'MG_NA', 'OA_OA', 'CL_SA',
                                                        'S_ZN',
                                                        'N_OA', 'C_SA', 'SA_ZN']
ligand_atom_types_list = ['A', 'C', 'CL', 'I', 'N', 'P', 'S', 'BR', 'HD', 'NA', 'F', 'OA', 'SA']
ligand_receptor_atom_type_pairs_electrostatic_list = ['I_N', 'OA_SA', 'FE_NA', 'HD_NA', 'A_CL', 'MG_SA', 'P_SA',
                                                        'C_NA',
                                                        'MN_NA', 'F_N', 'HD_N', 'HD_I', 'CL_MG', 'HD_S', 'CL_MN',
                                                        'F_OA',
                                                        'HD_OA', 'F_HD', 'A_SA', 'A_BR', 'BR_HD', 'SA_SA', 'A_MN',
                                                        'N_ZN',
                                                        'A_MG', 'I_OA', 'C_C', 'N_S', 'N_N', 'FE_N', 'NA_SA', 'BR_N',
                                                        'MN_N', 'A_P', 'BR_C', 'A_FE', 'MN_P', 'CL_OA', 'CU_HD',
                                                        'MN_S',
                                                        'A_S', 'FE_OA', 'NA_ZN', 'P_ZN', 'A_F', 'A_C', 'A_A', 'A_N',
                                                        'HD_MN', 'A_I', 'N_SA', 'C_OA', 'MG_P', 'BR_SA', 'CU_N',
                                                        'MN_OA',
                                                        'MG_N', 'HD_HD', 'C_FE', 'CL_NA', 'MG_OA', 'A_OA', 'CL_ZN',
                                                        'BR_OA', 'HD_ZN', 'HD_P', 'OA_P', 'OA_S', 'N_P', 'A_NA',
                                                        'CL_FE',
                                                        'HD_SA', 'C_MN', 'CL_HD', 'C_MG', 'FE_HD', 'MG_S', 'NA_S',
                                                        'NA_P',
                                                        'FE_SA', 'P_S', 'C_HD', 'A_ZN', 'CL_P', 'S_SA', 'CL_S',
                                                        'OA_ZN',
                                                        'N_NA', 'MN_SA', 'CL_N', 'NA_OA', 'F_ZN', 'C_ZN', 'HD_MG',
                                                        'C_F',
                                                        'C_I', 'C_CL', 'C_N', 'C_P', 'C_S', 'A_HD', 'F_SA', 'MG_NA',
                                                        'OA_OA', 'CL_SA', 'S_ZN', 'N_OA', 'C_SA', 'SA_ZN']
rotateable_bonds_count_list = ['rot_bonds']
active_site_flexibility_list = ['SIDECHAIN_OTHER', 'SIDECHAIN_ALPHA', 'BACKBONE_ALPHA', 'SIDECHAIN_BETA',
                                'BACKBONE_BETA', 'BACKBONE_OTHER']
hbonds_list = ['HDONOR-LIGAND_SIDECHAIN_BETA', 'HDONOR-LIGAND_BACKBONE_OTHER', 'HDONOR-LIGAND_SIDECHAIN_ALPHA',
                'HDONOR-RECEPTOR_SIDECHAIN_OTHER', 'HDONOR-RECEPTOR_BACKBONE_ALPHA',
                'HDONOR-RECEPTOR_SIDECHAIN_BETA',
                'HDONOR-RECEPTOR_SIDECHAIN_ALPHA', 'HDONOR-LIGAND_SIDECHAIN_OTHER', 'HDONOR-LIGAND_BACKBONE_BETA',
                'HDONOR-RECEPTOR_BACKBONE_BETA', 'HDONOR-RECEPTOR_BACKBONE_OTHER', 'HDONOR-LIGAND_BACKBONE_ALPHA']
hydrophobics_list = ['SIDECHAIN_OTHER', 'SIDECHAIN_ALPHA', 'BACKBONE_ALPHA', 'SIDECHAIN_BETA', 'BACKBONE_BETA',
                        'BACKBONE_OTHER']
stacking_list = ['ALPHA', 'BETA', 'OTHER']
pi_cation_list = ['LIGAND-CHARGED_BETA', 'LIGAND-CHARGED_ALPHA', 'RECEPTOR-CHARGED_BETA', 'RECEPTOR-CHARGED_OTHER',
                    'RECEPTOR-CHARGED_ALPHA', 'LIGAND-CHARGED_OTHER']
t_shaped_list = ['ALPHA', 'BETA', 'OTHER']
salt_bridges_list = ['ALPHA', 'BETA', 'OTHER']
# form columns
nnscore_header = vina_output_list + ['atp2_%s' % it for it in ligand_receptor_atom_type_pairs_less_than_two_half_list] \
                + ['atp4_%s' % it for it in ligand_receptor_atom_type_pairs_less_than_four_list] + ['lat_%s' % it for
                                                                                                    it in
                                                                                                    ligand_atom_types_list] \
                + ['ele_%s' % it for it in
                    ligand_receptor_atom_type_pairs_electrostatic_list] + rotateable_bonds_count_list + [
                    'siteflex_%s' % it for it in active_site_flexibility_list] \
                + ['hbond_%s' % it for it in hbonds_list] + ['hydrophobic_%s' % it for it in hydrophobics_list] + [
                    'stacking_%s' % it for it in stacking_list] \
                + ['pi_cation_%s' % it for it in pi_cation_list] + ['t_shaped_%s' % it for it in t_shaped_list] + [
                    'salt_bridges_%s' % it for it in salt_bridges_list]


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