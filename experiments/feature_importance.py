"""
Filename: feature_importance.py

Authors: Hakima Laribi

Description: This file defines the script that computes the feature importance given by the final ELSTM
trained on the whole learning set and tested on the holdout set using AdmDemoDx predictors.

"""
from tqdm import tqdm
import pickle
import os
import warnings

os.chdir('../')

if __name__ == '__main__':
    from src.models.lstm import HOMRBinaryLSTMC
    from src.data.processing.preparing import DataPreparer
    from src.data.processing.datasets import HOMRDataset
    from src.data.processing.constants import *
    from src.data.processing.sampling import OneShotSampler
    from src.utils.metric_scores import *
    from src.models.abstract_models.elstm_base_model import EnsembleLSTMBinaryClassifier
    import torch
    from src.models.abstract_models.lstm_base_models import LSTMBinaryClassifier
    from src.models.ensemble_lstm import HOMRBinaryELSTMC
    from settings.paths import Paths
    from typing import List, Dict
    from copy import deepcopy
    import pandas as pd
    import numpy as np


    def predict_from_lstm(x: torch.Tensor,
                          pretrainedlstm: LSTMBinaryClassifier):
        """
        Predicts from LSTM model

        Args:
            x: tensor with patients' data
            pretrainedlstm: LSTM model
        """
        # Set model for evaluation
        pretrainedlstm.eval()

        # Execute a forward pass and apply a sigmoid
        with torch.no_grad():
            if isinstance(x, list):
                return torch.cat([torch.sigmoid(pretrainedlstm(x[l].unsqueeze(dim=0))).cpu() for l in range(len(x))],
                                 dim=0).squeeze()
            else:
                return torch.sigmoid(pretrainedlstm(x).squeeze()).cpu()

    def predict_probas_and_compute_AUC(x_sizes: Dict[int, List[torch.tensor]],
                                       y_sizes: Dict[int, List[torch.tensor]],
                                       pretrainedmodel: EnsembleLSTMBinaryClassifier):
        """
        Computes a vectorized prediction of the ELSTM in patient's data permuted or not
        Args:
            x_sizes: dictionary with patients' data permuted or not, organized by the sequence length for
            vectorized prediction
            y_sizes: dictionary with patients' outcomes organized by the sequence length for
            vectorized prediction
            pretrainedmodel: Ensemble LSTM model
        """
        # 1- Build the dataset
        targets = np.empty(0, dtype=np.int64)
        pred = np.empty(0, dtype=np.float32)
        for size_, x in x_sizes.items():
            x_probas = []
            relative_indexes, _ = dataset.flatten_indexes(y_sizes[size_])
            y = torch.cat(y_sizes[size_])[relative_indexes]
            if size_ < MAX_VISIT:  # Case where we vectorize everything
                x = torch.stack(x)
                # Ensemble prediction
                for rank_model, pretrained_lstm in enumerate(pretrainedmodel._pretrained_models):
                    if rank_model >= size_:
                        probas = predict_from_lstm(x, pretrained_lstm._model)[relative_indexes]
                        x_probas.append(probas)

            else:
                pretrained_lstm = pretrainedmodel._pretrained_models[-1]
                probas = predict_from_lstm(x, pretrained_lstm._model)[relative_indexes]
                x_probas.append(probas)

            # Convert to numpy array
            x_probas = np.vstack(x_probas).T

            targets = np.concatenate((targets, y.cpu().detach().numpy()), axis=0)

            # 2- Prediction via Ensemble LSTM
            pred = np.concatenate((pred, np.mean(x_probas, axis=1)), axis=0)

        # 3- AUC computation
        auc = AUC()
        return auc(pred, targets)

    # Filter out the specific FutureWarning based on its category
    warnings.filterwarnings("ignore", category=FutureWarning)
    for task in ['oym']:
        dp = DataPreparer(task=task, train_file='csvs/dataset.csv', split_train_test=82104)

        # Get training and testing datasets then concatenate them
        df_train = dp.get__training_cohort
        df_test = dp.get__testing_cohort
        df_test = dp.create_CDSS_eligble_cohort(df_test)
        df = pd.concat([df_train, df_test]).reset_index(drop=True)

        print('### Read the dataset ###')
        # Dataset creation
        dataset = HOMRDataset(df,
                              task,
                              IDS,
                              cat_cols=CAT_COLS,
                              cont_cols=CONT_COLS,
                              encoding="one hot",
                              to_tensor=True,
                              temporal=True)

        # Initialize random state
        SEED = 101

        print('### Sampling ###')
        # Intern 5 cross validation sampling
        sampler = OneShotSampler(dataset=dataset, n_inner=0, valid_size=0.1)
        masks = sampler(learning_ids=df_train[IDS].unique(),
                        test_ids=df_test[IDS].unique(),
                        sampling_strategy=-1,
                        multiple_test_masks=False,
                        serialize=False)

        # Set the fixed hps
        # Load the pretrained models
        fixed_hps = {'max_epochs': 500,
                     'patience': 10,
                     'num_cont_col': len(dataset.cont_cols) + len(dataset.cat_cols),
                     'cat_idx': [],
                     'cat_sizes': []}
        lstm_model = HOMRBinaryLSTMC(**fixed_hps)
        pretrained_models = []
        for k in range(1, MAX_VISIT + 2):
            if k > MAX_VISIT:
                k = 'last'
            model_path = os.path.join(Paths.EXPERIMENTS_RECORDS,
                                      f"Holdout_LSTM_{k}_visits_AdmDemoDx",
                                      f"Split_0/torch_model.pt")
            pretrained_models += [deepcopy(lstm_model)]

        fixed_params = {'pretrained_models': pretrained_models}

        dataset.update_masks(train_mask=masks[0]['train'], valid_mask=masks[0]['valid'], test_mask=masks[0]['test'])

        # Load the pre-trained Super Learner model
        model = HOMRBinaryELSTMC(**fixed_params)
        ensemble_model_path = os.path.join(Paths.EXPERIMENTS_RECORDS,
                                           'Holdout_Bootstrapping_Ensemble_LSTMs_AdmDemoDx/Split_0/sklearn_model.sav')
        model_loaded = pickle.load(open(ensemble_model_path, 'rb'))

        model._model = model_loaded

        # Get all the features
        submasks = [str(lgt) for lgt in range(1, MAX_VISIT+1)] + ['other', 'all']
        features = dataset.cont_cols + dataset.cat_cols
        feature_scores_current = {feature: {sbmask: [] for sbmask in submasks} for feature in features}
        feature_scores_previous = {feature: {sbmask: [] for sbmask in submasks} for feature in features}
        bootstraps, n = 100, 1
        # Get the testing tensor
        x_test_original, y_test_original, idx_original = dataset[dataset.test_mask]
        primary_score = {submask: [] for submask in submasks}
        with tqdm(total=n * len(features) * bootstraps) as bar:
            for boot in range(bootstraps):
                # Sample with replacement
                n_patients = len(x_test_original)
                idx_patients = np.arange(len(x_test_original))
                np.random.seed(boot)
                sampled_indexes = np.random.choice(idx_patients, n_patients, replace=True).tolist()

                # Get the data and targets of sampled patients
                x_test = np.array(x_test_original, dtype='object')[sampled_indexes].tolist()
                y_test = np.array(y_test_original, dtype='object')[sampled_indexes].tolist()
                idx = np.array(idx_original, dtype='object')[sampled_indexes].tolist()

                # Rearrange in dictionaries according to history size
                y_sizes = {size_: [] for size_ in np.arange(MAX_VISIT + 1)}
                x_sizes = {size_: [] for size_ in np.arange(MAX_VISIT + 1)}
                idx_sizes = {size_: [] for size_ in np.arange(MAX_VISIT + 1)}
                x_test_stacked = torch.cat(x_test)
                y_test_stacked = torch.cat(y_test)
                init = 0
                for index in idx:
                    size = len(index) - 1
                    x_row = x_test_stacked[init:init + size + 1]
                    y_row = y_test_stacked[init:init + size + 1]
                    init += size + 1
                    size = size if size < MAX_VISIT else MAX_VISIT
                    x_sizes[size].append(x_row)
                    y_sizes[size].append(y_row)

                # Record the AUC score on the testing tensor
                for submask in submasks:
                    if submask == 'all':
                        x_sub = x_sizes
                        y_sub = y_sizes
                    elif submask == 'other':
                        size = MAX_VISIT
                        x_sub = {size: x_sizes[size]}
                        y_sub = {size: y_sizes[size]}
                    else:
                        size = int(submask) - 1
                        x_sub = {size: x_sizes[size]}
                        y_sub = {size: y_sizes[size]}

                    primary_score[submask].append(predict_probas_and_compute_AUC(x_sub, y_sub, model._model))

                # Stack the dataset
                x_stacked = torch.cat(x_test)
                y_stacked = torch.cat(y_test)

                # Get idx of current visits and idx of previous visits
                idx_current, _ = dataset.flatten_indexes(idx)
                idx_previous = [idx for idx in np.arange(x_stacked.shape[0]) if idx not in idx_current]

                # features = features[:4]
                for n_feature in range(len(features)):
                    scores_current = {submask: [] for submask in submasks}
                    scores_previous = {submask: [] for submask in submasks}
                    for k in range(n):
                        # Shuffle the dataset
                        xshuffled = deepcopy(x_stacked)
                        torch.manual_seed(boot + k)
                        permutations = torch.randperm(x_stacked.shape[0])
                        xshuffled[:, n_feature] = x_stacked[permutations, n_feature]

                        # Get the data of shuffled previous visits, and the dataset of shuffled current visits
                        xshuffled_previous, xshuffled_current = deepcopy(xshuffled), deepcopy(xshuffled)
                        xshuffled_previous[idx_previous, n_feature] = x_stacked[idx_previous, n_feature]
                        xshuffled_current[idx_current, n_feature] = x_stacked[idx_current, n_feature]

                        # Reshape the dataset for temporal analysis
                        init = 0
                        shuffled_current_hosp_sizes = {size_: [] for size_ in np.arange(MAX_VISIT+1)}
                        shuffled_previous_hosp_sizes = {size_: [] for size_ in np.arange(MAX_VISIT+1)}
                        y_sizes = {size_: [] for size_ in np.arange(MAX_VISIT+1)}

                        for idx_hosp in idx:
                            history_length = len(idx_hosp)
                            # Reassemble patient's history
                            patient_with_history_current = xshuffled_current[init:init + history_length]
                            patient_with_history_previous = xshuffled_previous[init:init + history_length]
                            patient_y = y_stacked[init:init + history_length]
                            init += history_length

                            # Store in the right dictionary according to its size
                            size = history_length - 1 if history_length <= MAX_VISIT else MAX_VISIT
                            shuffled_previous_hosp_sizes[size].append(patient_with_history_previous)
                            shuffled_current_hosp_sizes[size].append(patient_with_history_current)
                            y_sizes[size].append(patient_y)

                        # Compute score on the shuffled data
                        for submask in submasks:
                            if submask == 'all':
                                x_sub_curr = shuffled_current_hosp_sizes
                                x_sub_prev = shuffled_previous_hosp_sizes
                                y_sub = y_sizes
                            elif submask == 'other':
                                size = MAX_VISIT
                                x_sub_curr = {size: shuffled_current_hosp_sizes[size]}
                                x_sub_prev = {size: shuffled_previous_hosp_sizes[size]}
                                y_sub = {size: y_sizes[size]}
                            else:
                                size = int(submask) - 1
                                x_sub_curr = {size: shuffled_current_hosp_sizes[size]}
                                x_sub_prev = {size: shuffled_previous_hosp_sizes[size]}
                                y_sub = {size: y_sizes[size]}

                            scores_current[submask].append(
                                predict_probas_and_compute_AUC(x_sub_prev, y_sub, model._model))
                            # Score when the previous is shuffled
                            scores_previous[submask].append(
                                predict_probas_and_compute_AUC(x_sub_curr, y_sub, model._model))

                        bar.update()

                    for submask in submasks:
                        feature_scores_current[features[n_feature]][submask].append(np.mean(scores_current[submask]))
                        feature_scores_previous[features[n_feature]][submask].append(np.mean(scores_previous[submask]))
                    filepath_current = os.path.join(f"feature_scores_current_{task}")
                    filepath_previous = os.path.join(f"feature_scores_previous_{task}")
                    pickle.dump(feature_scores_current, open(filepath_current, "wb"))
                    pickle.dump(feature_scores_previous, open(filepath_previous, "wb"))

            for submask in submasks:

                columns = [f'AUC_BASE_{i + 1}' for i in range(bootstraps)]
                base_df = pd.DataFrame(columns=columns, index=features)
                for i in range(bootstraps):
                    base_df[columns[i]] = primary_score[submask][i]

                columns = [f'AUC_current_{i + 1}' for i in range(bootstraps)] + [f'AUC_previous_{i + 1}' for i in
                                                                                 range(bootstraps)]
                df = pd.DataFrame(columns=columns, index=features)

                for feature in features:
                    #
                    auc_scores_current = feature_scores_current[feature][submask]
                    auc_scores_previous = feature_scores_previous[feature][submask]
                    auc_scores = auc_scores_current + auc_scores_previous

                    df.loc[feature, columns] = auc_scores

                # Concatenate the DataFrames along the columns_to_anonymize (axis=1) to create the final DataFrame
                result_df = pd.concat([base_df, df], axis=1)
                result_df.reset_index(inplace=True)
                result_df.rename(columns={'index': 'Features'}, inplace=True)
                result_df.to_csv(f'feature_importance/feature_importance_{task}_{submask}.csv')
