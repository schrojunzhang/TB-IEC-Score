#!usr/bin/env python
# -*- coding:utf-8 -*-
# author:zhang xujun
# time = 2020-06-13

import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, confusion_matrix, cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler


class dataset(object):
    def __init__(self, name):
        self.target = name
        self.__path = os.path.dirname(os.path.realpath(__file__))
        self.__project_path = os.path.dirname(self.__path)
        self.__file_path = f'{self.__project_path}/source_data/3_combination'  # r'/home/xujun/Project_2/3_combination'
        self.__vina_file_path = f'{self.__project_path}/source_data/3_combination'
        self.__score_path = f'{self.__project_path}/source_data/4_score'  # r'/home/xujun/Project_2/4_score'
        self.__name_path = r'{}/{}'.format(self.__score_path, name)
        self.__csv = '{}/{}/{}.csv'.format(self.__file_path, self.target, self.target)
        self.__vina_csv = '{}/{}/{}.csv'.format(self.__vina_file_path, self.target, self.target)
        self.vdw_dic = {'sp': ['r_i_glide_evdw_x'],
                        'cation': ['r_glide_XP_PiCat', 'r_glide_XP_PiStack'],
                        'autodock': ['vdw'],
                        'galaxy': ['vdw_pl', 'vdw_l'],
                        'xscore': ['VDW'],
                        'goldscore': ['Goldscore.Internal.Vdw'],
                        'vina': ['gauss1', 'gauss2', 'repulsion'],
                        'smina': ['gauss(o=0,_w=0.3,_c=8)', 'gauss(o=0.5,_w=0.3,_c=8)', 'gauss(o=1,_w=0.3,_c=8)',
                                  'gauss(o=1.5,_w=0.3,_c=8)', 'gauss(o=2,_w=0.3,_c=8)', 'gauss(o=2.5,_w=0.3,_c=8)',
                                  'gauss(o=0,_w=0.5,_c=8)',
                                  'gauss(o=1,_w=0.5,_c=8)', 'gauss(o=2,_w=0.5,_c=8)', 'gauss(o=0,_w=0.7,_c=8)',
                                  'gauss(o=1,_w=0.7,_c=8)',
                                  'gauss(o=2,_w=0.7,_c=8)', 'gauss(o=0,_w=0.9,_c=8)', 'gauss(o=1,_w=0.9,_c=8)',
                                  'gauss(o=2,_w=0.9,_c=8)',
                                  'gauss(o=3,_w=0.9,_c=8)', 'gauss(o=0,_w=1.5,_c=8)', 'gauss(o=1,_w=1.5,_c=8)',
                                  'gauss(o=2,_w=1.5,_c=8)',
                                  'gauss(o=3,_w=1.5,_c=8)', 'gauss(o=4,_w=1.5,_c=8)', 'gauss(o=0,_w=2,_c=8)',
                                  'gauss(o=1,_w=2,_c=8)',
                                  'gauss(o=2,_w=2,_c=8)', 'gauss(o=3,_w=2,_c=8)', 'gauss(o=4,_w=2,_c=8)',
                                  'gauss(o=0,_w=3,_c=8)',
                                  'gauss(o=1,_w=3,_c=8)', 'gauss(o=2,_w=3,_c=8)', 'gauss(o=3,_w=3,_c=8)',
                                  'gauss(o=4,_w=3,_c=8)',
                                  'repulsion(o=0.4,_c=8)', 'repulsion(o=0.2,_c=8)', 'repulsion(o=0,_c=8)',
                                  'repulsion(o=-0.2,_c=8)',
                                  'repulsion(o=-0.4,_c=8)', 'repulsion(o=-0.6,_c=8)', 'repulsion(o=-0.8,_c=8)',
                                  'repulsion(o=-1,_c=8)']}
        self.hbond_dic = {'autodock': ['hb'],
                          'galaxy': ['hbond_pl', 'hbond_l'],
                          'vina': ['HB_x'],
                          'sp': ['r_i_glide_hbond'],
                          'xscore': ['HB_y'],
                          'goldscore': ['Goldscore.Internal.HBond', 'Goldscore.External.HBond'],
                          'chemscore': ['Chemscore.Hbond'],
                          'plp': ['PLP.part.hbond', 'PLP.Chemscore.Hbond', 'Chemscore.CHOScore'],
                          'smina': ['non_dir_h_bond(g=-0.7,_b=0,_c=8)', 'non_dir_h_bond(g=-0.7,_b=0.2,_c=8)',
                                    'non_dir_h_bond(g=-0.7,_b=0.5,_c=8)', 'non_dir_h_bond(g=-1,_b=0,_c=8)',
                                    'non_dir_h_bond(g=-1,_b=0.2,_c=8)', 'non_dir_h_bond(g=-1,_b=0.5,_c=8)',
                                    'non_dir_h_bond(g=-1.3,_b=0,_c=8)', 'non_dir_h_bond(g=-1.3,_b=0.2,_c=8)',
                                    'non_dir_h_bond(g=-1.3,_b=0.5,_c=8)']}
        self.elec = {
            'autodock': ['qq'],
            'sp': ['r_i_glide_ecoul_x'],
            'xp': ['r_i_glide_ecoul_y', 'r_glide_XP_Electro'],
            'galaxy': ['qq_pl', 'qq_l'],
            'smina': ['electrostatic(i=1,_^=100,_c=8)', 'electrostatic(i=2,_^=100,_c=8)'],
            'nnscore': ['ele_%s' % it for it in
                        ['I_N', 'OA_SA', 'FE_NA', 'HD_NA', 'A_CL', 'MG_SA', 'P_SA', 'C_NA', 'MN_NA',
                         'F_N', 'HD_N', 'HD_I', 'CL_MG', 'HD_S', 'CL_MN', 'F_OA', 'HD_OA', 'F_HD',
                         'A_SA', 'A_BR', 'BR_HD', 'SA_SA', 'A_MN', 'N_ZN', 'A_MG', 'I_OA', 'C_C',
                         'N_S', 'N_N', 'FE_N', 'NA_SA', 'BR_N', 'MN_N', 'A_P', 'BR_C', 'A_FE',
                         'MN_P', 'CL_OA', 'CU_HD', 'MN_S', 'A_S', 'FE_OA', 'NA_ZN', 'P_ZN', 'A_F',
                         'A_C', 'A_A', 'A_N', 'HD_MN', 'A_I', 'N_SA', 'C_OA', 'MG_P', 'BR_SA',
                         'CU_N', 'MN_OA', 'MG_N', 'HD_HD', 'C_FE', 'CL_NA', 'MG_OA', 'A_OA', 'CL_ZN',
                         'BR_OA', 'HD_ZN', 'HD_P', 'OA_P', 'OA_S', 'N_P', 'A_NA', 'CL_FE', 'HD_SA',
                         'C_MN', 'CL_HD', 'C_MG', 'FE_HD', 'MG_S', 'NA_S', 'NA_P', 'FE_SA', 'P_S',
                         'C_HD', 'A_ZN', 'CL_P', 'S_SA', 'CL_S', 'OA_ZN', 'N_NA', 'MN_SA', 'CL_N',
                         'NA_OA', 'F_ZN', 'C_ZN', 'HD_MG', 'C_F', 'C_I', 'C_CL', 'C_N', 'C_P',
                         'C_S', 'A_HD', 'F_SA', 'MG_NA', 'OA_OA', 'CL_SA', 'S_ZN', 'N_OA', 'C_SA',
                         'SA_ZN']]}
        self.polar = {
            'affini': ['Polar_Component_Term'],
            'xp': ['r_glide_XP_Sitemap']}
        self.hydrophobic = {
            'chemscore': ['Chemscore.Lipo'],
            'affini': ['Hydrophobic_Complementarity_Term'],
            'xscore': ['HP', 'HM', 'HS'],
            'xp': ['r_glide_XP_LipophilicEvdW', 'r_glide_XP_PhobEn', 'r_glide_XP_ClBr'],
            'vina': ['hydrophobic'],
            'smina': ['hydrophobic(g=0.5,_b=1.5,_c=8)', 'hydrophobic(g=0.5,_b=1,_c=8)', 'hydrophobic(g=0.5,_b=2,_c=8)',
                      'hydrophobic(g=0.5,_b=3,_c=8)']}
        self.non_hydrophobic = {
            'smina': ['non_hydrophobic(g=0.5,_b=1.5,_c=8)']
        }
        self.entropy = {
            'autodock': ['tors'],
            'vina': ['rbond'],
            'smo': ['SMoG2016_Rotor', 'SMoG2016_lnMass'],
            'chemscore': ['Chemscore.Rot'],
            'xp': ['r_glide_XP_LowMW', 'r_glide_XP_RotPenal']
            # 'affini': []
        }
        self.desolv = {
            'autodock': ['dsolv'],
            'smina': ['ad4_solvation(d-sigma=3.6,_s/q=0.01097,_c=8)', 'ad4_solvation(d-sigma=3.6,_s/q=0,_c=8)'],
            'galaxy': ['desolv_pl', 'desolv_l']
            # 'affini': []
        }
        self.knowledge = {
            'dsx': ['atom_pairs', 'intra_clashes', 'sas_score'],
            'drug': ['DrugScore'],
            'KBP': ['SMoG2016_KBP2016'],
            'plp': ['PLP.PLP'],
            'statescore': ['ASP.Map']
        }
        self.clash = {
            'plp': ['PLP.ligand.clash'],
            'chemscore': ['Chemscore.DEClash'],
            'asp': ['ASP.DEClash']
        }
        self.metal = {
            'chemscore': ['Chemscore.Metal'],
            'sp': ['r_i_glide_metal']
        }
        self.torsion = {
            'chemscore': ['Chemscore.DEInternal'],
            'goldscore': ['Goldscore.Internal.Torsion'],
            'asp': ['ASP.DEInternal']
        }
        self.nn_score = {
            'colse_contact': ['atp2_%s' % it for it in ['A_MN', 'OA_SA', 'HD_N', 'N_ZN', 'A_MG', 'HD_NA', 'A_CL',
                                                        'MG_OA', 'FE_HD', 'A_OA', 'NA_ZN', 'A_N', 'C_OA', 'F_HD',
                                                        'C_HD',
                                                        'NA_SA', 'A_ZN', 'C_NA', 'N_N', 'MN_N', 'F_N', 'FE_OA', 'HD_I',
                                                        'BR_C', 'MG_NA', 'C_ZN', 'CL_MG', 'BR_OA', 'A_FE', 'CL_OA',
                                                        'CL_N', 'NA_OA', 'F_ZN', 'HD_P', 'CL_ZN', 'C_C', 'C_CL', 'FE_N',
                                                        'HD_S', 'HD_MG', 'C_F', 'A_NA', 'BR_HD', 'HD_OA', 'HD_MN',
                                                        'A_SA', 'A_F', 'HD_SA', 'A_C', 'A_A', 'F_SA', 'C_N', 'HD_ZN',
                                                        'OA_OA', 'N_SA', 'CL_FE', 'C_MN', 'CL_HD', 'OA_ZN', 'MN_OA',
                                                        'C_MG', 'F_OA', 'CD_OA', 'S_ZN', 'N_OA', 'C_SA', 'N_NA', 'A_HD',
                                                        'HD_HD', 'SA_ZN']],
            'semi_contact': ['atp4_%s' % it for it in ['I_N', 'OA_SA', 'FE_NA', 'HD_NA', 'A_CL', 'MG_SA', 'A_CU',
                                                       'P_SA', 'C_NA', 'MN_NA', 'F_N', 'HD_N', 'HD_I', 'CL_MG', 'HD_S',
                                                       'CL_MN', 'F_OA', 'HD_OA', 'F_HD', 'A_SA', 'A_BR', 'BR_HD',
                                                       'SA_SA', 'A_MN', 'N_ZN', 'A_MG', 'I_OA', 'C_C', 'N_S', 'N_N',
                                                       'FE_N', 'NA_SA', 'BR_N', 'MN_N', 'A_P', 'BR_C', 'A_FE', 'MN_P',
                                                       'CL_OA', 'CU_HD', 'MN_S', 'A_S', 'FE_OA', 'NA_ZN', 'P_ZN', 'A_F',
                                                       'A_C', 'A_A', 'A_N', 'HD_MN', 'A_I', 'N_SA', 'C_OA', 'MG_P',
                                                       'BR_SA', 'CU_N', 'MN_OA', 'MG_N', 'HD_HD', 'C_FE', 'CL_NA',
                                                       'MG_OA', 'A_OA', 'CL_ZN', 'BR_OA', 'HD_ZN', 'HD_P', 'OA_P',
                                                       'OA_S', 'N_P', 'A_NA', 'CL_FE', 'HD_SA', 'C_MN', 'CL_HD', 'C_MG',
                                                       'FE_HD', 'MG_S', 'NA_S', 'NA_P', 'FE_SA', 'P_S', 'C_HD', 'A_ZN',
                                                       'CL_P', 'S_SA', 'CL_S', 'OA_ZN', 'N_NA', 'MN_SA', 'CL_N',
                                                       'NA_OA', 'C_ZN', 'C_CD', 'HD_MG', 'C_F', 'C_I', 'C_CL', 'C_N',
                                                       'C_P', 'C_S', 'A_HD', 'F_SA', 'MG_NA', 'OA_OA', 'CL_SA', 'S_ZN',
                                                       'N_OA', 'C_SA', 'SA_ZN']]}
        self.energy_types = {
            'vdw': self.vdw_dic,
            'hb': self.hbond_dic,
            'elec': self.elec,
            'polar': self.polar,
            'lipo': self.hydrophobic,
            'entropy': self.entropy,
            'desolv': self.desolv,
            'knowledge': self.knowledge,
            'clash': self.clash,
            'metal': self.metal,
            'torsion': self.torsion,
            'non_lipo': self.non_hydrophobic,
            'nnscore': self.nn_score,
        }  # 全体能量项
        self.__combine_energy_types = {
            'vdw': self.vdw_dic,
            'hb': self.hbond_dic,
            'elec': self.elec,
            'lipo': self.hydrophobic,
            'entropy': self.entropy,
            'clash': self.clash
        }  # 用于组合的能量项

    def preprocess(self, *, train_x, test_x, scale=True, variance_filter=True,
                   normalization=True):  # data preprocessing
        # scale
        if scale:
            scaler = StandardScaler().fit(train_x)
            train_x = scaler.transform(train_x)
            test_x = scaler.transform(test_x)
        # variance_filter
        if variance_filter:
            threshold = VarianceThreshold().fit(train_x)
            train_x = threshold.transform(train_x)
            test_x = threshold.transform(test_x)
        # normalization
        if normalization:
            normalizer = Normalizer(norm='l2').fit(train_x)
            train_x = normalizer.transform(train_x)
            test_x = normalizer.transform(test_x)
        return train_x, test_x

    def get_item(self, *csvs, energy_type='vdw', mode='diy', return_df=False, get_name=False, get_class=False,
                 not_vina=True):
        # if parameters are allowed
        if not mode in ('diy', 'all', 'help') and not energy_type in self.energy_types:
            print('parameter error')
        else:
            # get energy terms
            energy_item = self.energy_types[energy_type]
            # print help
            if mode == 'help':
                return '''energy_type = {}\ncsv_type = {}\nmode = {}\nreturn_df = {}\nget_name = {}\nget_class = {}'''.format \
                    ([i for i in self.energy_types], [i for i in energy_item], ('diy', 'all', 'help'), [True, False],
                     [True, False], [True, False])
            # get all energy terms
            elif mode == 'all':
                csvs = [i for i in energy_item]
            else:
                pass
            # get columns
            __columns = []
            temp = [__columns.extend([i for i in energy_item[j]]) for j in csvs]
            # wether get ligand name
            if get_name:
                __columns = ['name']  # get name
            elif get_class:
                __columns = ['class']  # get class
            else:
                # if csv is input
                if not csvs:
                    print('need csv parameter')
                    sys.exit()  # exit
            # if return df
            if return_df == False:  # do not return
                return __columns  # return columns
            else:  # return data
                # if get descriptors of vina pose
                if not_vina:
                    df = pd.read_csv(self.__csv, encoding='utf-8')

                else:  # get descriptors of vina pose
                    df = pd.read_csv(self.__vina_csv, encoding='utf-8')
                # get corresponding data
                vdw_data = df.loc[:, __columns].dropna()
                return vdw_data

    def test_metric(self, *, y_true, y_pred, cal_confusion=True, cal_acc=True, cal_f1=True, cal_mcc=True,
                    cal_kappa=True):
        # init
        result = []
        # cal_confusion
        if cal_confusion:
            confu_matrix = confusion_matrix(y_true, y_pred)
            result.extend(confu_matrix)
            # accuracy
        if cal_acc:
            acc = accuracy_score(y_true, y_pred)
            result.append(acc)
            # f1score
        if cal_f1:
            f1 = f1_score(y_true, y_pred)
            result.append(f1)
            # mcc
        if cal_mcc:
            mcc = matthews_corrcoef(y_true, y_pred)
            result.append(mcc)
            # cohen_kappa
        if cal_kappa:
            kappa = cohen_kappa_score(y_true, y_pred)
            result.append(kappa)
        return result

    def cal_importance(self, energy_type='vdw', not_vina=True):
        # check parameters
        if energy_type not in self.__combine_energy_types:
            raise ValueError
        # combine df based on energy type
        # if descriptors are calculated from vina pose
        if not_vina:
            df_feature = self.get_item(energy_type=energy_type, mode='all', return_df=True)
        else:
            df_feature = self.get_item(energy_type=energy_type, mode='all', return_df=True, not_vina=False)
            # merge df
        df_all = pd.concat([self.get_item(get_name=True, return_df=True), df_feature,
                            self.get_item(get_class=True, return_df=True)], axis=1, ignore_index=True)
        # drop nan
        df = df_all.dropna()
        #  get x and y
        x = df.iloc[:, 1:-1]
        y = df.iloc[:, -1]
        # split dataset
        train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.8, random_state=42)
        # data preprocessing
        train_x, test_x = self.preprocess(train_x=train_x, test_x=test_x, variance_filter=False)
        # cal importance score
        clf = RandomForestClassifier(n_jobs=24)
        clf.fit(train_x, train_y)
        # merge energy types and their importance score
        importance = dict(zip(df_feature.columns.tolist(), clf.feature_importances_))
        # get the sum of importance score of each scoring function
        sf_importance = []  # init
        for sf in self.__combine_energy_types[
            energy_type]:  # get scoring function {'cation': ['r_glide_XP_PiCat', 'r_glide_XP_PiStack']}
            sf_energy_terms = self.__combine_energy_types[energy_type][
                sf]  # get the energy terms of specific energy type from each scoring function ['r_glide_XP_PiCat', 'r_glide_XP_PiStack']
            sf_importance.append((np.mean([float(importance[term]) for term in
                                           sf_energy_terms])))  # cal importance score of each scoring function
        # output importance score {'sp':0.5}
        importance = dict(zip(self.__combine_energy_types[energy_type], sf_importance))
        return importance

    def out_report(self, *, algo, vdw_order, hb_order, elec_order, lipo_order, entropy_order, clash_order,
                   cross_score, tn, fp, fn, tp, acc, f1, mcc, kappa):
        # defined file name
        csv_file = '{}/{}.csv'.format(self.__name_path, algo)
        # create file
        if not os.path.exists(csv_file):
            pd.DataFrame(['name', 'cross_score', 'tn', 'fp', 'fn', 'tp', 'acc', 'f1', 'mcc', 'kappa']).T.to_csv(
                csv_file, index=False, header=False)  # 写入标题栏
        # get combination name
        combination = '{}_{}_{}_{}_{}_{}'.format(vdw_order, hb_order, elec_order, lipo_order, entropy_order,
                                                 clash_order)
        df_1 = pd.DataFrame([combination, cross_score, tn, fp, fn, tp, acc, f1, mcc, kappa]).T  # get score
        # output to csv file
        df_1.to_csv(csv_file, index=False, header=False, mode='a')

    def get_ifp_data(self, target, sf='plp'):
        # sf = asp, chemscore, goldscore, plp
        # file path
        csv_file = '{}/{}/{}_IFP.csv'.format(self.__file_path, target, sf)
        # read data
        df = pd.read_csv(csv_file, encoding='utf-8')
        return df


if __name__ == '__main__':
    import warnings
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt

    warnings.filterwarnings('ignore')
    import seaborn as sns

    scaler = MinMaxScaler()
    decomposer = PCA(n_components=1)
    clusterr = KMeans(n_clusters=3)
    targets = ['akt1', 'ampc', 'cp3a4', 'cxcr4', 'gcr', 'hivpr', 'hivrt', 'kif11']
    cmp = ['blue', 'red', 'green', 'grey', 'purple', 'pink', 'yellow']
    markers = ['.', '<', '>', '^', 'v', 's', 'p', 'o', ',']
    for idx, i in enumerate(['vdw', 'elec', 'hb', 'clash', 'lipo', 'entropy']):
        y_ = 0
        fig, ax = plt.subplots(figsize=(10, 10))
        feats = []
        targetss = []
        sfs = []
        print(f'----------------------------energy type: {i}  ----------------------------')
        for k in targets:
            test = dataset(k)
            y_ += 1
            df = test.get_item(energy_type=i, mode='all', return_df=True)
            energy_dic = {}
            for j, v in test.energy_types[i].items():
                tmp_feat = df.loc[:, v].values
                tmp_feat = scaler.fit_transform(tmp_feat)
                tmp_feat = decomposer.fit_transform(tmp_feat)  # .reshape((-1))
                tmp_feat = tmp_feat.mean()
                feats.append(tmp_feat)
                targetss.append(k)
                sfs.append(j)
        #         plt.scatter(x=[tmp_feat], y=[0], alpha=0.4, label=f'{i}_{k}_{j}'
        #                     # , c=cmp[y_]
        #                     , marker=markers[y_]
        #                     )

        feats = np.array(feats).reshape((-1, 1))
        clusters = clusterr.fit_predict(feats)
        clusters = np.array(clusters).reshape((-1, 1)).astype(int)
        targetss = np.array(targetss).reshape((-1, 1))
        sfs = np.array(sfs).reshape((-1, 1))
        data = np.concatenate([feats, clusters, targetss, sfs], axis=-1)
        df = pd.DataFrame(data, columns=['value', 'cluster', 'target', 'sf'])
        print(df)
        sns.scatterplot(x='value', y=[0] * 64, data=df, hue='cluster')
        # plt.legend()
        plt.title(f'{i}')
        plt.show()
        break
