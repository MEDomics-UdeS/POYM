"""
Filename: global_model_variation.py

Authors: Hakima Laribi

Description: This file defines the script that computes the model's performance variation, when shuffling each feature
simultaneously for current and previous features.

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
    from src.models.abstract_models.eln_base_model import EnsembleLongitudinalNetworkBinaryClassifier
    import torch
    from src.models.abstract_models.lstm_base_models import LSTMBinaryClassifier
    from src.models.eln import HOMRBinaryELNC
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
                                       pretrainedmodel: EnsembleLongitudinalNetworkBinaryClassifier):
        """
        Computes a vectorized prediction of the ELN in patient's data permuted or not
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
    task = 'oym'
    for test_type in ['last', 'any']:
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

        # Get the corresponding test mask
        test_mask = masks[0]['test'] if test_type == 'last' else masks[0]['test_random_visit']

        dataset.update_masks(train_mask=masks[0]['train'], valid_mask=masks[0]['valid'], test_mask=test_mask)

        # Instantiate the ELN
        model = HOMRBinaryELNC(**fixed_params)

        # Get all the features
        submasks = ['all']
        features = dataset.cont_cols + dataset.cat_cols
        feature_scores = {feature: {sbmask: [] for sbmask in submasks} for feature in features}
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

                    primary_score[submask].append(predict_probas_and_compute_AUC(x_sub, y_sub, model._model))

                # Stack the dataset
                x_stacked = torch.cat(x_test)
                y_stacked = torch.cat(y_test)

                # Get idx of current visits and idx of previous visits
                idx_current, _ = dataset.flatten_indexes(idx)
                idx_previous = [idx for idx in np.arange(x_stacked.shape[0]) if idx not in idx_current]

                for n_feature in range(len(features)):
                    scores_shuffled = {submask: [] for submask in submasks}
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
                        shuffled_hosp_sizes = {size_: [] for size_ in np.arange(MAX_VISIT + 1)}
                        y_sizes = {size_: [] for size_ in np.arange(MAX_VISIT + 1)}

                        for idx_hosp in idx:
                            history_length = len(idx_hosp)
                            # Reassemble patient's history
                            patient_with_history = xshuffled[init:init + history_length]
                            patient_y = y_stacked[init:init + history_length]
                            init += history_length

                            # Store in the right dictionary according to its size
                            size = history_length - 1 if history_length <= MAX_VISIT else MAX_VISIT
                            shuffled_hosp_sizes[size].append(patient_with_history)
                            y_sizes[size].append(patient_y)

                        for submask in submasks:
                            if submask == 'all':
                                x_sub = shuffled_hosp_sizes
                                y_sub = y_sizes
                            scores_shuffled[submask].append(predict_probas_and_compute_AUC(x_sub, y_sub, model._model))

                        bar.update()

                    for submask in submasks:
                        feature_scores[features[n_feature]][submask].append(np.mean(scores_shuffled[submask]))
                    filepath_aucs = os.path.join(f"vi_feature_scores_shuffled_{task}")
                    pickle.dump(feature_scores, open(filepath_aucs, "wb"))

            for submask in submasks:

                columns = [f'AUC_BASE_{i + 1}' for i in range(bootstraps)]
                base_df = pd.DataFrame(columns=columns, index=features)
                for i in range(bootstraps):
                    base_df[columns[i]] = primary_score[submask][i]

                columns = [f'AUC_shuffled_{i + 1}' for i in range(bootstraps)]
                df = pd.DataFrame(columns=columns, index=features)

                for feature in features:
                    #
                    auc_scores = feature_scores[feature][submask]

                    df.loc[feature, columns] = auc_scores

                # Concatenate the DataFrames along the columns_to_anonymize (axis=1) to create the final DataFrame
                result_df = pd.concat([base_df, df], axis=1)
                result_df.reset_index(inplace=True)
                result_df.rename(columns={'index': 'Features'}, inplace=True)
                result_df.to_csv(f'feature_importance/performance_variation_{task}_{submask}_{test_type}.csv')
