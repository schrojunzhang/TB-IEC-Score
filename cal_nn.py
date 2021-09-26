#!usr/bin/env python
# -*- coding:utf-8 -*-
# used for cal nnscore
import os  # pandas as pd,
from multiprocessing import Pool

from NNScore2module import PDB, binana, command_line_parameters


def nn_score(ligand):
    lig_name = ligand.split('.')[0]
    to_ligand = '%s/%s' % (path_name_lig, ligand)
    ligand_pdbqt = '%s/%s.pdbqt' % (path_name_pdbqt, lig_name)
    ligand_pdbqt_pred = '%s/%s_pre.pdbqt' % (path_name, ligand)
    # prepare ligand
    log_file = '%s/%s_nn.txt' % (path_name, lig_name)  # 分数文件
    if not os.path.exists(ligand_pdbqt):
        cmdline = 'module purge &&'
        cmdline += 'module load vina &&'
        cmdline += 'prepare_ligand4.py -l %s -o %s -A hydrogens ' % (to_ligand, ligand_pdbqt)
        os.system(cmdline)

    if not os.path.exists(ligand_pdbqt_pred):
        with open(ligand_pdbqt, 'r')as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            if line.startswith('ATOM'):
                new_lines.append(line[:23] + '   ' + line[26:])
            else:
                new_lines.append(line)
        new_lig = ''.join(new_lines)
        with open(ligand_pdbqt_pred, 'w')as f:
            f.write(new_lig)
    # commandline
    cmd = "/home/xujun/Soft/SCORE_Function/NNscore/NNScore2module.py -receptor %s -ligand %s" % (
    protein_pred, ligand_pdbqt_pred)
    try:
        params_list = cmd.split()
        cmd_params = command_line_parameters(params_list)
        receptor = PDB()
        receptor.LoadPDB_from_file(protein_pred)
        receptor.OrigFileName = protein_pred
        d = binana(ligand_pdbqt_pred, receptor, cmd_params, "", "", "")

        result = [
                     lig_name] + d.vina_output + d.ligand_receptor_atom_type_pairs_less_than_two_half.values() + d.ligand_receptor_atom_type_pairs_less_than_four.values() \
                 + d.ligand_atom_types.values() + d.ligand_receptor_atom_type_pairs_electrostatic.values() + d.rotateable_bonds_count.values() \
                 + d.active_site_flexibility.values() + d.hbonds.values() + d.hydrophobics.values() + d.stacking.values() + d.pi_cation.values() \
                 + d.t_shaped.values() + d.salt_bridges.values()
    except:
        result = [lig_name] + [0 for i in range(len(header_list))]
    with open(log_file, 'w')as f:
        f.write(str(result))
    # collect result
    cmd = 'module load anaconda3/5.1.0 &&'
    cmd += 'python /home/xujun/temp/nn_write.py %s %s' % (log_file, csv_file)
    os.system(cmd)
    # pd.DataFrame(result).T.to_csv(csv_file, index=None, header=None, mode='a')
    # remove file
    os.remove(ligand_pdbqt_pred)


def cal_nn():
    global path
    path = r'/home/xujun/Project_2/2_descriptor'
    names = ['akt1', 'ampc', 'cp3a4', 'cxcr4', 'gcr', 'hivpr', 'hivrt', 'kif11',
             'KAT2A', 'MAPK1', 'MTORC1', 'VDR', 'PKM2', 'TP53', 'ALDH1', 'ESR1_ant', 'GBA', 'FEN1']
    names = ['akt1', 'cp3a4', 'cxcr4', 'gcr', 'hivpr', 'hivrt', 'kif11',
             'KAT2A', 'MAPK1', 'MTORC1', 'VDR', 'PKM2', 'TP53', 'ALDH1', 'ESR1_ant', 'GBA', 'FEN1']
    for name in names:
        global path_name, path_name_lig, protein_pred, csv_file, path_name_pdbqt
        path_name = '%s/%s' % (path, name)
        protein_file = '%s/%s_protein.pdb' % (path_name, name)
        protein_pre = '%s/%s_vina_p.pdb' % (path_name, name)
        protein_pred = '%s/%s_vina_p2.pdbqt' % (path_name, name)
        crystal_src_file = '%s/%s_crystal_ligand.mol2' % (path_name, name)
        crystal_file = '%s/%s_crystal_ligand.pdbqt' % (path_name, name)
        # move file and convert struct
        if not os.path.exists(crystal_file):
            cmdline = 'module purge &&'
            cmdline += 'module load vina &&'
            cmdline += 'prepare_ligand4.py -l %s -o %s -A hydrogens ' % (crystal_src_file, crystal_file)
            os.system(cmdline)
        # prepare protein
        if not os.path.exists(protein_pred):
            cmdline = 'cat %s | sed \'/HETATM/\'d > %s &&' % (protein_file, protein_pre)
            cmdline += 'module purge &&'
            cmdline += 'module load vina &&'
            cmdline += 'prepare_receptor4.py -r %s -o %s -A hydrogens -U nphs_lps_waters_nonstdres &&' % (
                protein_pre, protein_pred)
            cmdline += 'rm  %s' % protein_pre
            os.system(cmdline)
        # write csv file
        header_log = '%s/header.txt' % path_name
        header = str(['name'] + header_list)
        with open(header_log, 'w')as f:
            f.write(header)
        csv_file = '%s/nnscore.csv' % path_name
        cmd = 'module load anaconda3/5.1.0 &&'
        cmd += 'python /home/xujun/temp/nn_write.py %s %s' % (header_log, csv_file)
        os.system(cmd)
        if not os.path.exists(csv_file):
            pd.DataFrame(['name'] + header_list).T.to_csv(csv_file, index=None, header=None)
        # get ligands
        path_name_lig = '%s/lig' % path_name
        ligands = os.listdir(path_name_lig)
        # create pdbqt filefold
        path_name_pdbqt = '%s/lig_pdbqt' % path_name
        if not os.path.exists(path_name_pdbqt):
            os.mkdir(path_name_pdbqt)
        # multicore
        pool = Pool(24)
        pool.map(nn_score, ligands)
        pool.close()
        pool.join()


if __name__ == '__main__':
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
    header_list = vina_output_list + ['atp2_%s' % it for it in ligand_receptor_atom_type_pairs_less_than_two_half_list] \
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
    cal_nn()
