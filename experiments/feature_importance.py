import os

from tqdm import tqdm

os.chdir('../')

if __name__ == '__main__':
    from src.models.lstm import HOMRBinaryLSTMC
    from src.data.processing.preparing import DataPreparer
    from src.data.processing.datasets import HOMRDataset
    from src.data.processing.constants import *
    from src.data.processing.sampling import SimpleSampler
    from src.utils.metric_scores import *
    import torch
    from src.models.abstract_models.lstm_base_models import LSTMBinaryClassifier
    from typing import List
    from copy import deepcopy
    import pandas as pd
    import numpy as np
    from re import search
    import matplotlib.pyplot as plt


    def predict_probas_and_compute_AUC(x: List[torch.tensor],
                                       y: List[torch.tensor],
                                       mask: List[List[int]],
                                       pretrainedmodel: LSTMBinaryClassifier):
        # Set model for evaluation
        pretrainedmodel.eval()

        # Execute a forward pass and apply a sigmoid
        with torch.no_grad():
            pred = torch.cat([torch.sigmoid(pretrainedmodel(x[l].unsqueeze(dim=0))).cpu() for l in range(len(x))],
                             dim=0).squeeze()

        y = torch.cat(y)

        # Flatten mask
        relative_indexes = []
        initial_index = 0
        for m, indexes in enumerate(mask):
            relative_indexes.append(initial_index + len(indexes) - 1)
            initial_index += len(indexes)

        y = y[relative_indexes]
        pred = pred[relative_indexes]

        auc = AUC()
        return auc(pred, y)


    def reshape_dataset(x: torch.tensor,
                        y: torch.tensor,
                        idx: List[List[int]], ):
        initial_id = 0
        reshaped_x = []
        reshaped_y = []
        # Put x and y in the same shapes as idx
        for _, indexes in enumerate(idx):
            last_id = len(indexes) if isinstance(indexes, list) else 1
            reshaped_x.append(x[initial_id:initial_id + last_id])
            reshaped_y.append(y[initial_id:initial_id + last_id])
            initial_id += last_id
        return reshaped_x, reshaped_y, idx


    task = 'oym'
    # Prepare HOMR learning data
    dp = DataPreparer(task=task, test_fie='csvs/df_holdout.csv')

    # Get training and testing datasets then concatenate them
    df_train = dp.get__training_cohort
    df_test = dp.get__testing_cohort
    df_test = dp.create_CDSS_eligble_cohort(df_test)
    df = pd.concat([df_train, df_test]).reset_index(drop=True)

    print('### Read the dataset ###')
    # Dataset creation
    dataset = HOMRDataset(df, task, IDS, cat_cols=CAT_COLS, cont_cols=CONT_COLS, encoding="one hot", to_tensor=True,
                          temporal=True)

    # Initialize random state
    SEED = 101

    print('### Sampling ###')
    # Intern 5 cross validation sampling and split learnet set to training and validation
    sampler = SimpleSampler(dataset=dataset, n_inner=0, valid_size=0.1)
    masks = sampler(learning_ids=np.array(df_train[IDS].unique()),
                    test_ids=np.array(df_test[IDS].unique()),
                    sampling_strategy=-1,
                    temporal_sampling=True,
                    multiple_test_masks=False)

    fixed_params = {'max_epochs': 500,
                    'patience': 10,
                    'num_cont_col': len(dataset.cont_cols) + len(dataset.cat_cols),
                    'cat_idx': [],
                    'cat_sizes': [],
                    'cat_emb_sizes': []}

    dataset.update_masks(train_mask=masks[0]['train'], valid_mask=masks[0]['valid'], test_mask=masks[0]['test'])

    # Load the pre-trained model
    model = HOMRBinaryLSTMC(**fixed_params)
    # Path to the pretrained LSTM model
    model_path = 'records_exp/oym/LSTM_tuned_holdout_AdmDemo/Split_0/torch_model.pt'
    model._model = torch.load(model_path)

    # Get all the features
    feature_scores = {}
    features = dataset.cont_cols + dataset.cat_cols

    # Get the training tensor
    x_test, y_test, idx = dataset[dataset.test_mask]

    # Record the AUC score on the training tensor
    primary_score = predict_probas_and_compute_AUC(x_test, y_test, idx, model._model)

    xdep = torch.cat(x_test)

    n = 10
    with tqdm(total=n * len(features)) as bar:
        for n_feature in range(len(features)):
            scores = 0
            for k in range(n):
                # Shuffle the dataset
                xshuffled = deepcopy(xdep)
                permutations = torch.randperm(xdep.shape[0])
                xshuffled[:, n_feature] = xdep[permutations, n_feature]
                # Reshape the dataset for temporal analysis
                x_reshaped, y_reshaped, idx = reshape_dataset(xshuffled, torch.cat(y_test), idx)
                # Compute score on the shuffled
                scores += predict_probas_and_compute_AUC(x_reshaped, y_reshaped, idx, model._model)
                bar.update()

            feature_scores[features[n_feature]] = scores / n

    differences = {feature: primary_score - score for feature, score in feature_scores.items()}
    sum_of_differences = sum(differences.values())
    # Step 4: Compute the ratio of each score metric's difference to the sum of differences
    ratios = {feature: diff / sum_of_differences for feature, diff in differences.items()}

    # Create DataFrames from the dictionaries
    differences_df = pd.DataFrame.from_dict(differences, orient='index', columns=["Differences"])
    ratios_df = pd.DataFrame.from_dict(ratios, orient='index', columns=["Ratios"])
    scores_df = pd.DataFrame.from_dict(feature_scores, orient='index', columns=["AUCs"])

    # Concatenate the DataFrames along the columns (axis=1) to create the final DataFrame
    result_df = pd.concat([scores_df, differences_df, ratios_df], axis=1)
    result_df.reset_index(inplace=True)
    result_df.rename(columns={'index': 'Features'}, inplace=True)
    result_df.to_csv('feature_importance.csv')

    # Compute feature importance for each group of features
    feature_importance = {'Demographics': 0,
                          'Admission characteristics': 0,
                          'Comorbidity diagnoses': 0,
                          'Admission diagnoses': 0}

    for index, row in result_df.iterrows():
        feature = row["Features"]
        ratio = row["Ratios"]
        if search("^(age.*)|^(gender.*)", feature):
            feature_importance['Demographics'] += ratio
        elif search("^(adm_.*)", feature):
            feature_importance['Admission diagnoses'] += ratio
        elif search("^(dx_.*)|(has_dx)", feature):
            feature_importance['Comorbidity diagnoses'] += ratio
        else:
            feature_importance['Admission characteristics'] += ratio

    # Plot feature importance
    # Customizing the plot
    fig, ax = plt.subplots(figsize=(8, 3.5))  # Adjust the figure size as needed

    # Specify a custom color palette (you can change the colors as desired)
    colors = ['skyblue', 'lightgreen', 'salmon', 'gold']

    # Convert the feature importance to percentage and plot the horizontal bar chart
    width = 0.7
    y_axis, x_axis = np.array(list(feature_importance.keys())), np.array(list(feature_importance.values()))
    bars = ax.barh(y_axis, x_axis * 100, height=width, align='center', color=colors)

    # Set the font size for the axis labels and tick labels
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xlabel('Feature Importance Percentage', fontsize=16)

    plt.tight_layout()
    plt.show()
