"""
Filename: final_validation.py

Authors: Hakima Laribi

Description: This file defines the script that performs the final validation step, it performs an evaluation
             on patients eligible for GOC discussions

"""

from copy import deepcopy
import json
import os
from typing import List

import numpy as np
import torch

os.chdir('../')


def masks_for_bootstrapping(n_boot: int,
                            test_patient_ids: List[int]):
    """
    Create n_boot of patient's ids for bootstrapping
    Args:
        n_boot: the number of bootsraps
        test_patient_ids: list of patient's ids'
    """
    np.random.seed(0)
    sampled_ids = [np.random.choice(test_patient_ids, len(test_patient_ids), replace=True).tolist() for _ in
                   range(n_boot)]

    return sampled_ids

if __name__ == '__main__':
    from src.data.processing.preparing import DataPreparer
    from src.data.processing.datasets import HOMRDataset
    from src.data.processing.constants import *
    from src.data.processing.sampling import OneShotSampler
    from src.utils.metric_scores import *
    from src.evaluating.evaluation import Evaluator
    from src.models.lstm import HOMRBinaryLSTMC
    from src.models.ensemble_lstm import HOMRBinaryELSTMC
    from settings.paths import Paths
    from hps.sanity_check_hps import RNN_HPS

    task = 'oym'
    # Prepare HOMR learning data
    dp = DataPreparer(task=task, test_fie='csvs/df_holdout.csv')
    df_train = dp.get__training_cohort
    df_test = dp.get__testing_cohort # Use these patients to test the temporal validity of the ELSTM
    # Get the cohort eligible for GOC discussions
    df_test = dp.create_CDSS_eligble_cohort(df_test)
    df = pd.concat([df_train, df_test]).reset_index(drop=True)

    # Initialize random state
    SEED = 101

    # Initialization of the dictionary containing the evaluation metrics
    evaluation_metrics = [BinaryAccuracy(),
                          BinaryBalancedAccuracy(),
                          BinaryBalancedAccuracy(Reduction.GEO_MEAN),
                          Sensitivity(),
                          Specificity(),
                          AUPRC(),
                          BrierScore(),
                          Precision(),
                          NegativePredictiveValue(),
                          NFN(),
                          NFP(),
                          NTN(),
                          NTP(),
                          F2Score(),
                          F1Score(),
                          BinaryBalancedAccuracy(),
                          AUC()]

    for i, cat_columns in enumerate([CAT_COLS, OTHER_CAT_COL]):
        if i == 0:
            exp_suffix = 'AdmDemoDx'
            print('AdmDemoDx experiments')
        else:
            exp_suffix = 'AdmDemo'
            print('AdmDemo experiments')

        print("### Evaluate using LSTM ###")
        print('### Read the dataset ###')
        # Dataset creation
        dataset = HOMRDataset(df, task, IDS, CONT_COLS, cat_columns, encoding="one hot", temporal=True, to_tensor=True)

        print('### Sampling ###')
        # Inner nested 5 cross validation sampling
        sampler = OneShotSampler(dataset=dataset, n_inner=5, valid_size=0.1)
        masks = sampler(learning_ids=np.array(df_train[IDS].unique()),
                        test_ids=np.array(df_test[IDS].unique()),
                        sampling_strategy=-1,
                        multiple_test_masks=False,
                        serialize=False)

        def update_fixed_params(subset, itr):
            return {'max_epochs': 500,
                    'patience': 10,
                    'num_cont_col': len(subset.cont_cols) + len(subset.cat_cols),
                    'cat_idx': [],
                    'cat_sizes': []}


        print("### Optimization of hyperparameters for LSTM_last ###")

        evaluator = Evaluator(model_constructor=HOMRBinaryLSTMC,
                              dataset=dataset,
                              masks=masks,
                              hps=RNN_HPS,
                              n_trials=100,
                              evaluation_metrics=evaluation_metrics,
                              fixed_params_update_function=update_fixed_params,
                              evaluation_name=f"Holdout_LSTM_tuned_{exp_suffix}",
                              save_hps_importance=True,
                              save_optimization_history=True,
                              seed=SEED)

        evaluator.evaluate()

        print("### Train each single LSTM constituting the ELSTM with the selected hyperparameters")

        # Get the selected hyperparameters
        def update_fixed_params(subset, itr):
            file_hps = os.path.join(Paths.EXPERIMENTS_RECORDS, f'Holdout_LSTM_tuned_{exp_suffix}')
            with open(os.path.join(file_hps, f'Split_{itr}', 'records.json'), "r") as file:
                data_file = json.load(file)
            hps = {'max_epochs': 500,
                   'patience': 10,
                   'num_cont_col': len(subset.cont_cols) + len(subset.cat_cols),
                   'cat_idx': [],
                   'cat_sizes': []}
            for hps_name, hps_value in data_file['hyperparameters'].items():
                hps[hps_name] = hps_value

            return hps


        for LSTM_i in range(1, MAX_VISIT + 2):
            sampling_strategy = LSTM_i if LSTM_i < MAX_VISIT + 1 else -1
            samples = str(LSTM_i) if LSTM_i < MAX_VISIT + 1 else 'last'
            print(f" Train LSTM_{samples} ..")

            # Split the dataset according to the sampling strategy of the corresponding LSTM
            sampler = OneShotSampler(dataset=dataset, n_inner=0, valid_size=0.1)
            masks = sampler(learning_ids=np.array(df_train[IDS].unique()),
                            test_ids=np.array(df_test[IDS].unique()),
                            sampling_strategy=sampling_strategy,
                            multiple_test_masks=False,
                            serialize=False)

            """
               Evaluator validation with LSTM_i
            """

            evaluator = Evaluator(model_constructor=HOMRBinaryLSTMC,
                                  dataset=dataset,
                                  masks=masks,
                                  hps={},
                                  n_trials=0,
                                  evaluation_metrics=evaluation_metrics,
                                  fixed_params_update_function=update_fixed_params,
                                  evaluation_name=f"Holdout_LSTM_{str(samples)}_visits_{exp_suffix}",
                                  save_hps_importance=True,
                                  save_optimization_history=True,
                                  seed=SEED)

            evaluator.evaluate()

        # Prediction using the ELSTM
        n_bootstraps = 100
        print(f" Prediction with the ELSTM with {n_bootstraps} bootstrap samples ..")

        # Split the dataset
        sampler = OneShotSampler(dataset=dataset, n_inner=0, valid_size=0.1)
        masks = sampler(learning_ids=np.array(df_train[IDS].unique()),
                        test_ids=masks_for_bootstrapping(100, df_test[IDS].unique()),
                        sampling_strategy=-1,
                        multiple_test_masks=False,
                        serialize=False)


        # Define the hyperparameters of the ELSTM
        def update_fixed_params(subset, itr):
            fixed_hps = {'num_cont_col': len(subset.cont_cols) + len(subset.cat_cols),
                         'cat_idx': [],
                         'cat_sizes': [],}
            # Load the pretrained models
            lstm_model = HOMRBinaryLSTMC(**fixed_hps)
            pretrained_models = []
            for k in range(1, MAX_VISIT + 2):
                if k > MAX_VISIT:
                    k = 'last'
                model_path = os.path.join(Paths.EXPERIMENTS_RECORDS,
                                          f"Holdout_LSTM_{k}_visits_{exp_suffix}",
                                          f"Split_0/torch_model.pt")
                lstm_model._model = torch.load(model_path)
                pretrained_models += [deepcopy(lstm_model)]

            return {'pretrained_models': pretrained_models}


        """
           Evaluator validation with ELSTM
        """
        evaluator = Evaluator(model_constructor=HOMRBinaryELSTMC,
                              dataset=dataset,
                              masks=masks,
                              hps=RNN_HPS,
                              n_trials=0,
                              evaluation_metrics=evaluation_metrics,
                              fixed_params_update_function=update_fixed_params,
                              evaluation_name=f"Holdout_Bootstrapping_Ensemble_LSTMs_{exp_suffix}",
                              save_hps_importance=True,
                              save_optimization_history=True,
                              seed=SEED)

        evaluator.evaluate()
