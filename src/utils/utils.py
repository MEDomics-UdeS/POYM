"""
Filename: utils.py

Authors: Hakima Laribi

Description: This file defines som utilities methods

"""
import json
import os
from re import search
from typing import List

import numpy as np
from numpy.random import choice
import pandas as pd
from pandas import DataFrame
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

def generate_random_variables(df_to_anonymize: pd.DataFrame):
    """
    Use a categorical distribution to generate random data for each column in the dataset

    Args:
         df_to_anonymize: dataset with the original data to use to generate random data
    """
    random_dict = {}
    for col in df_to_anonymize.columns:
        # Get the categories appearing probabilities of each categorical column
        categories_probabilities = df_to_anonymize[col].value_counts(normalize=True)

        # Generate n data for each column respecting the categories proportions in the original dataset
        random_dict[col] = choice(a=categories_probabilities.index,
                               size=df_to_anonymize.shape[0],
                               p=categories_probabilities.values)

    return random_dict
def generate_randomized_dataset(df: pd.DataFrame,
                                columns_to_anonymize: List[str],
                                target_column: str):
    """
    Generate random data for each visit in the dataset following a categorical distribution

    Args:
        df: dataset with the original visits
        columns_to_anonymize: columns in the dataset to be anonymized
        target_column: name of the column with the outcomes
    """
    # Sort dataset by patient_id and order of occurrence of each visit
    df['admission_date'] = pd.to_datetime(df['admission_date'])
    df.sort_values(by=['patient_id','admission_date'], inplace=True, ignore_index=True)

    # Get indexes of positive and negative outcomes
    outcome_indexes =  df.groupby(target_column).apply(lambda x: x.index.tolist()).to_dict()

    # Map each patient_id to a new generated id, and replace the original patient_ids in the dataset
    patient_map =  dict(zip(df['patient_id'].unique(), np.arange(len(df['patient_id'].unique()))))
    df['patient_id'] = df['patient_id'].map(patient_map.get)

    random_dataset = []

    for target in df[target_column].unique():
        df_target = df[df[target_column] == target]

        # Anonymize all columns that are relative to the visit
        visit_cols_anonymize = [col for col in columns_to_anonymize if col != 'age_original' and col != 'gender']

        # Join the numerical and categorical data generated
        random_visits =  DataFrame.from_dict({**generate_random_variables(df_target[visit_cols_anonymize])})

        # Add target column and visit_id
        random_visits['visit_id'] = outcome_indexes[target]
        random_visits[target_column] = [target] * df_target.shape[0]
        random_dataset.append(random_visits)

    # Concatenate the dataset generated for each outcome
    random_dataset = pd.concat(random_dataset)

    # Reorder according to the order of appearance of positive/negative outcomes in the original dataset
    random_dataset.sort_values(by='visit_id', inplace=True, ignore_index=True)

    # Add patient_ids, so each patient has the exact number of visits as in the original dataset, and each visit has
    # the same outcome
    random_dataset['patient_id'] = df['patient_id']

    # Generate patient specific feature: age at first visit and gender
    random_dataset['age_original'] = [-1] * random_dataset.shape[0]
    random_dataset['gender'] = [-1] * random_dataset.shape[0]
    patient_dataset = df.groupby('patient_id')[['age_original', 'gender']].first().reset_index()[['age_original', 'gender']]
    patient_dict = generate_random_variables(patient_dataset)

    # Add age variable and gender at each visit
    patient_indexes = df.groupby('patient_id').apply(lambda x: x.index.tolist()).to_dict()
    patient_ages = []
    patient_genders = []
    for i, (patient_id, indexes) in enumerate(patient_indexes.items()):
        # Generate a random shifting for each visit
        age_at_first_visit = patient_dict['age_original'][i]
        shifting = [0] + sorted(choice(a=np.arange(0, 11), size=len(indexes) - 1).tolist())
        patient_ages += [age_at_first_visit + shift for shift in shifting]
        patient_genders += [patient_dict['gender'][i] ]* len(indexes)

    random_dataset['age_original'] = patient_ages
    random_dataset['gender'] = patient_genders

    # Reorder columns and save in csv
    reordered_columns = ['patient_id', 'visit_id'] + columns_to_anonymize + [target_column]
    random_dataset[reordered_columns].to_csv('csvs/random_dataset.csv', index=False)







