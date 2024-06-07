"""
Filename: model_selection.py

Authors: Hakima Laribi

Description: This file defines the script that performs the model selection step, it compares the ELSTM, BLSTM and RF
             on patients of the learning set

"""
from copy import deepcopy
import json
import os

import torch

os.chdir('../')

if __name__ == '__main__':
    from hps.sanity_check_hps import RNN_HPS, RGF_HPS
    from settings.paths import Paths
    from src.models.ranger_forest import HOMRBinaryRGFC
    from src.data.processing.preparing import DataPreparer
    from src.data.processing.datasets import HOMRDataset
    from src.data.processing.constants import *
    from src.data.processing.sampling import KFoldsSampler
    from src.utils.metric_scores import *
    from src.utils.statistical_tests import compute_significant_difference
    from src.evaluating.evaluation import Evaluator
    from src.models.lstm import HOMRBinaryLSTMC
    from src.models.ensemble_lstm import HOMRBinaryELSTMC

    task = 'oym'
    # Prepare HOMR learning data
    dp = DataPreparer(task=task)
    df = dp.get__training_cohort

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
            print('AdmDemoDx experiments ..')
        else:
            exp_suffix = 'AdmDemo'
            print('AdmDemo experiments ..')


        print('### Read the dataset ###')
        # Dataset creation
        dataset = HOMRDataset(df, task, IDS, CONT_COLS, cat_columns, encoding="one hot", temporal=True, to_tensor=True)

        print('### Sampling ###')
        # Nested 5 cross validation sampling
        sampler = KFoldsSampler(dataset=dataset, valid_size=0.1, k=5, inner_k=5)
        masks = sampler(sampling_strategy=-1, multiple_test_masks=False, serialize=False)


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
                              evaluation_name=f"LSTM_tuned_{exp_suffix}",
                              save_hps_importance=True,
                              save_optimization_history=True,
                              seed=SEED)

        evaluator.evaluate()

        print("### Train each single LSTM constituting the ELSTM with the selected hyperparameters")

        # Get the selected hyperparameters
        def update_fixed_params(subset, itr):
            file_hps = os.path.join(Paths.EXPERIMENTS_RECORDS, f'LSTM_tuned_{exp_suffix}')
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
            sampler = KFoldsSampler(dataset=dataset, valid_size=0.1, k=5, inner_k=0)
            masks = sampler(sampling_strategy=sampling_strategy, multiple_test_masks=True, serialize=False)

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
                                  evaluation_name=f"LSTM_{str(samples)}_visits_{exp_suffix}",
                                  save_hps_importance=True,
                                  save_optimization_history=True,
                                  seed=SEED)

            evaluator.evaluate()

        # Prediction using the ELSTM
        print(f" Prediction with the ELSTM ..")

        # Split the dataset
        sampler = KFoldsSampler(dataset=dataset, valid_size=0.1, k=5, inner_k=0)
        masks = sampler(sampling_strategy=-1, multiple_test_masks=True, serialize=False)

        # Define the hyperparameters of the ELSTM
        def update_fixed_params(subset, itr):
            fixed_hps = {'num_cont_col': len(subset.cont_cols) + len(subset.cat_cols),
                         'cat_idx': [],
                         'cat_sizes': []}
            # Load the pretrained models
            lstm_model = HOMRBinaryLSTMC(**fixed_hps)
            pretrained_models = []
            for k in range(1, MAX_VISIT + 2):
                if k > MAX_VISIT:
                    k = 'last'
                model_path = os.path.join(Paths.EXPERIMENTS_RECORDS,
                                          f"LSTM_{k}_visits_{exp_suffix}",
                                          f"Split_{itr}/torch_model.pt")
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
                              evaluation_name=f"Ensemble_LSTMs_{exp_suffix}",
                              save_hps_importance=True,
                              save_optimization_history=True,
                              seed=SEED)

        evaluator.evaluate()

        print("### Evaluate using Random Forest ###")
        print('### Read the dataset ###')
        # Dataset creation
        dataset = HOMRDataset(df, task, IDS, CONT_COLS+['nb_visits'], cat_columns, encoding="one hot")

        # Initialize random state
        SEED = 101

        print('### Sampling ###')
        # Nested 5 cross validation sampling
        sampler = KFoldsSampler(dataset=dataset, valid_size=0., k=5, inner_k=5)
        masks = sampler(sampling_strategy=None, multiple_test_masks=True, serialize=False)

        """
           Evaluator validation with ranger forest
        """

        evaluator = Evaluator(model_constructor=HOMRBinaryRGFC,
                              dataset=dataset,
                              masks=masks,
                              hps=RGF_HPS,
                              n_trials=100,
                              evaluation_metrics=evaluation_metrics,
                              evaluation_name=f"RandomForest_tuned_{exp_suffix}",
                              save_hps_importance=True,
                              save_optimization_history=True,
                              seed=SEED)

        evaluator.evaluate()

        print("### Evaluate using BLSTM ###")
        print('### Read the dataset ###')
        # Dataset creation
        dataset = HOMRDataset(df,
                              task,
                              IDS,
                              cat_cols=cat_columns,
                              cont_cols=CONT_COLS + ['nb_visits'],
                              encoding="one hot",
                              to_tensor=True,
                              temporal=True)

        # Initialize random state
        SEED = 101

        print('### Sampling ###')
        # Nested 5 cross validation sampling
        sampler = KFoldsSampler(dataset=dataset, valid_size=0.1, k=5, inner_k=5)
        masks = sampler(sampling_strategy=None, multiple_test_masks=True, serialize=True)

        def update_fixed_params(subset, itr):
            return {'max_epochs': 500,
                    'patience': 10,
                    'num_cont_col': len(subset.cont_cols) + len(subset.cat_cols),
                    'cat_idx': [],
                    'cat_sizes': []}

        """
           Evaluator validation with BLSTM
        """

        evaluator = Evaluator(model_constructor=HOMRBinaryLSTMC,
                              dataset=dataset,
                              masks=masks,
                              hps=RNN_HPS,
                              n_trials=100,
                              evaluation_metrics=evaluation_metrics,
                              fixed_params_update_function=update_fixed_params,
                              evaluation_name=f"BLSTM_tuned_{exp_suffix}",
                              save_hps_importance=True,
                              save_optimization_history=True,
                              seed=SEED)

        evaluator.evaluate()

        print('Wilcoxon test to measure the difference significance')
        # ELSTM vs RF
        compute_significant_difference(model_1=os.path.join(Paths.EXPERIMENTS_RECORDS, f"Ensemble_LSTMs_{exp_suffix}"),
                                       model_2=os.path.join(Paths.EXPERIMENTS_RECORDS, f"RandomForest_tuned_{exp_suffix}"),
                                       n_splits=5,
                                       saving_dir=os.path.join(Paths.RECORDS,
                                                               'p_values',
                                                               f"ELSTM_vs_RF_{exp_suffix}"))
        # ELSTM vs BLSTM
        compute_significant_difference(model_1=os.path.join(Paths.EXPERIMENTS_RECORDS, f"Ensemble_LSTMs_{exp_suffix}"),
                                       model_2=os.path.join(Paths.EXPERIMENTS_RECORDS, f"BLSTM_tuned_{exp_suffix}"),
                                       n_splits=5,
                                       saving_dir=os.path.join(Paths.RECORDS,
                                                               'p_values',
                                                               f"ELSTM_vs_BLSTM_{exp_suffix}"))
