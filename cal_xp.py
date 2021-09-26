#!usr/bin/env python
# -*- coding:utf-8 -*-
import os
# used for cal glide sp and glide xp


def cal_glide_xp(name, node, id=''):
    if id:
        path_local_id = '%s/%s/%s' % (path, name, id)
    else:
        path_local_id = '%s/%s' % (path, name)
        id = name

    protein_file = '%s/%s_protein.pdb' % (path_local_id, id)
    ligand_file = '%s/%s_ligand.mol2' % (path_local_id, id)
    grid_file = '%s/glide-grid.zip' % path_local_id
    in_file = '%s/XP.in' % path_local_id
    lin = '''GRIDFILE   %s
LIGANDFILE   %s
POSES_PER_LIG   1
POSE_OUTTYPE   ligandlib
##POSTDOCK   False
DOCKING_METHOD   mininplace
PRECISION   XP
WRITE_XP_DESC   True
WRITE_CSV True
        ''' % (grid_file, ligand_file)
    with open(in_file, 'w') as f:
        f.write(lin)

    cmdline = 'cd %s &&' % path_local_id
    cmdline += 'module load schrodinger &&'
    cmdline += 'glide XP.in -HOST cu0%s:24' % node
    os.system(cmdline)


def cal_dudue_xp():
    path = r'/home/xujun/Project_2/download'
    # targets
    # names = ['KAT2A', 'MAPK1', 'MTORC1', 'OPRK1', 'VDR', 'PKM2', 'PPARG', 'TP53', 'ADRB2', 'ALDH1',
    #          'ESR1_ant', 'GBA', 'IDH1', 'FEN1'] #'VDR',
    names_dude = ['ampc', 'cp3a4', 'cxcr4', 'gcr', 'hivpr', 'hivrt', 'kif11']
    names = [[x] for x in names_dude]
    # names_big_than_50 = ['KAT2A', 'MAPK1', 'MTORC1', 'VDR', 'PKM2', 'TP53', 'ALDH1',
    #          'ESR1_ant', 'GBA', 'FEN1']
    # for name in names_big_than_50:
    #     path_name = '%s/%s' % (path, name)
    #     pdb_ids = os.listdir(path_name)
    #     for i in pdb_ids:
    #         name_id = [name, i]
    #         names.append(name_id)
    # # submit job
    for i, name in enumerate(names):
        # get node
        node = divmod(i, 8)[1] + 1
        if node == 2:
            node = 8
        if len(name) == 1:
            cal_glide_xp(name[0], node=node)
        else:
            cal_glide_xp(name[0], id=name[1], node=node)


def cal_lt_xp():
    # global path
    path = r'/home/xujun/Project_2/2_descriptor'
    names = ['KAT2A', 'MAPK1', 'MTORC1', 'VDR', 'PKM2', 'TP53', 'ALDH1', 'ESR1_ant', 'GBA', 'FEN1']
    for i, name in enumerate(names):
        # get note
        node = divmod(i, 8)[1] + 1
        # submit job
        cal_glide_xp(name, node=node)
