"""
Filename: statistical_tests.py

Authors: Hakima Laribi

Description: This file defines the method that perform statistical tests to compute the significance
             of difference between the predictions of two models

"""
import json
import os
from re import search
import numpy as np
from scipy.stats import wilcoxon


def compute_significant_difference(model_1: str,
                                   model_2: str,
                                   n_splits: int,
                                   saving_dir: str
                                   ):
    """
    Computes the significant difference between the AUROCs of two models
    Args:
        model_1: path to the directory where the experiment of model 1 is recorded
        model_2: path to the directory where the experiment of model 2 is recorded
        n_splits: number of splits used in both experiments, must be identical
        saving_dir: path to the directory where to save the computed p-values
    """
    # Create saving directory
    os.makedirs(saving_dir)
    p_values = {}

    # Iterate over all the splits
    for i in range(n_splits):
        # Read the data recorded at each split for both models
        with open(os.path.join(model_1, 'Split_' + str(i), 'records.json'), "r") as file:
            model_1_data = json.load(file)

        with open(os.path.join(model_2, 'Split_' + str(i), 'records.json'), "r") as file:
            model_2_data = json.load(file)

        for section in model_1_data.keys():
            # Compare AUCs of testing sets only
            if search("(test).*(metrics)$", section):
                if section not in p_values.keys():
                    p_values[section] = {}

                model_1_auc, model_2_auc = float(model_1_data[section]['AUC']), float(model_2_data[section]['AUC'])

                # Save AUCs
                p_values[section][str(i)] = {'auc': [model_1_auc, model_2_auc]}

    for section, p_val_section in p_values.items():
        auc_1 = np.array([float(p_val_section[str(m)]['auc'][0]) for m in range(n_splits)])
        auc_2 = np.array([float(p_val_section[str(m)]['auc'][1]) for m in range(n_splits)])

        p_values[section]['wilcoxon'] = wilcoxon(auc_1, auc_2, alternative='greater')[1]
        p_values[section]['wilcoxon_less'] = wilcoxon(auc_1, auc_2, alternative='less')[1]

    with open(os.path.join(saving_dir, 'p_values.json'), "w") as file:
        json.dump(p_values, file, indent=True)
