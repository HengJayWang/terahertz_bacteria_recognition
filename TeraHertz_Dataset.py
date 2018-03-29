"""
TeraHertz_Dataset.py

2017.12.05 Heng-Jie Wang

Description: Read the feature vector file (.csv) by Numpy and Pandas

for training set

"""

import pandas as pd
import numpy as np


def load_dataset():
    for index in range(2, 9):
        ecoli_file_name = "acesolution_E.Coli_2017_1102/E_Coli_10_" + str(index) +\
                          "_order_10_ul_x5_feature_vector_data.csv"
        yeast_file_name = "acesolution_Yeast_2017_1109/Yeast_10_" + str(index) +\
                          "_order_10ul_x5_feature_vector_data.csv"

        ecoli_csv_file = pd.read_csv(ecoli_file_name, low_memory=False)
        yeast_csv_file = pd.read_csv(yeast_file_name, low_memory=False)

        if index == 2:
            ecoli_feature = ecoli_csv_file.values[:, 1:].transpose()
            yeast_feature = yeast_csv_file.values[:, 1:].transpose()
        else:
            ecoli_feature = np.concatenate([ecoli_feature,
                                            ecoli_csv_file.values[:, 1:].transpose()])
            yeast_feature = np.concatenate([yeast_feature,
                                            yeast_csv_file.values[:, 1:].transpose()])

    rol, _ = ecoli_feature.shape

    ecoli_target = np.zeros(rol)
    yeast_target = np.ones(rol)

    feature = np.concatenate([ecoli_feature, yeast_feature])
    target = np.concatenate([ecoli_target, yeast_target])

    return feature, target
