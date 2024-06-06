"""
Filename: recording.py

Authors: Hakima Laribi
         Nicolas Raymond

Description: This file is used to define the Recorder class

"""

import json
import os
import pickle
from collections import Counter
from typing import Any, Dict, List, Union, Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy import arange, max, median, min, std
from torch import tensor, save, zeros
from torch.nn import Module

from settings.paths import Paths
from src.data.processing.constants import *
from src.data.processing.sampling import MaskType
from src.data.processing.datasets import HOMRDataset
from src.models.abstract_models.base_models import BinaryClassifier
from src.recording.constants import *
from src.utils.metric_scores import *


class Recorder:
    """
    Recorder objects used save results of the experiments
    """
    # Dictionary that associate the mask types to their proper section
    MASK_TO_SECTION = {METRICS: {},
                       RESULTS: {}
                       }

    def __init__(self,
                 evaluation_name: str,
                 index: int,
                 recordings_path: str,
                 masks_types: List[str]):
        """
        Sets protected attributes

        Args:
            evaluation_name: name of the evaluation
            index: index of the outer split
            recordings_path: path leading to where we want to save the results
        """

        # We store the protected attributes
        self._data = {NAME: evaluation_name,
                      INDEX: index,
                      DATA_INFO: {},
                      HYPERPARAMETERS: {},
                      HYPERPARAMETER_IMPORTANCE: {},
                      COEFFICIENT: {}}

        # Update the dictionary that associates the mask types to their proper section
        for section in [METRICS, RESULTS]:
            for mask_type in masks_types:
                if mask_type is not MaskType.INNER:
                    mask_section = mask_type + '_' + section
                    self._data[mask_section] = {}
                    self.MASK_TO_SECTION[section][mask_type] = mask_section

        self._path = os.path.join(recordings_path, evaluation_name, f"Split_{index}")

        # We create the folder where the information will be saved
        os.makedirs(self._path, exist_ok=True)

    def generate_file(self) -> None:
        """
        Save the protected dictionary into a json file

        Returns: None
        """
        # We save all the data collected in a json file
        filepath = os.path.join(self._path, RECORDS_FILE)
        with open(filepath, "w") as file:
            json.dump(self._data, file, indent=True)

    def record_coefficient(self,
                           name: str,
                           value: float) -> None:
        """
        Saves the value associated to a coefficient (used for linear regression)

        Args:
            name: name of the variable associated to the coefficient
            value: value of the coefficient

        Returns: None
        """
        self._data[COEFFICIENT][name] = value

    def record_data_info(self,
                         data_name: str,
                         data: Any) -> None:
        """
        Records the specific value "data" associated to the variable "data_name" in
        the protected dictionary

        Args:
            data_name: name of the variable for which we want to save a specific information
            data: value we want to store

        Returns: None

        """
        self._data[DATA_INFO][data_name] = data

    def record_hyperparameters(self, hyperparameters: Dict[str, Any]) -> None:
        """
        Saves the hyperparameters in the protected dictionary

        Args:
            hyperparameters: dictionary of hyperparameters and their value

        Returns: None
        """
        # We save all the hyperparameters
        for key in hyperparameters.keys():
            self._data[HYPERPARAMETERS][key] = round(hyperparameters[key], 6) if \
                isinstance(hyperparameters[key], float) else hyperparameters[key]

    def record_hyperparameters_importance(self, hyperparameter_importance: Dict[str, float]) -> None:
        """
        Saves the hyperparameters' importance in the protected dictionary

        Args:
            hyperparameter_importance: dictionary of hyperparameters and their importance

        Returns: None
        """
        # We save all the hyperparameters importance
        for key in hyperparameter_importance.keys():
            self._data[HYPERPARAMETER_IMPORTANCE][key] = round(hyperparameter_importance[key], 4)

    def record_model(self, model: BinaryClassifier) -> None:
        """
        Saves a model using pickle or torch's save function

        Args:
            model: model to save

        Returns: None

        """
        # If the model is a torch module with save it using torch
        if isinstance(model, Module):
            save(model, os.path.join(self._path, "model.pt"))
        else:
            # We save the model with pickle
            filepath = os.path.join(self._path, "model.sav")
            pickle.dump(model, open(filepath, "wb"))

    def record_scores(self,
                      score: float,
                      metric: str,
                      mask_type: str = MaskType.TRAIN) -> None:
        """
        Saves the score associated to a metric

        Args:
            score: float
            metric: name of the metric
            mask_type: train, test or valid

        Returns: None
        """
        # We find the proper section name
        section = Recorder.MASK_TO_SECTION[METRICS][mask_type]

        # We save the score of the given metric
        self._data[section][metric] = round(score, 6)

    def record_predictions(self,
                           ids: List[str],
                           predictions: tensor,
                           probabilities: tensor,
                           targets: Optional[tensor],
                           mask_type: str = MaskType.TRAIN) -> None:
        """
        Save the predictions of a given model for each patient ids

        Args:
            ids: patient/participant ids
            predictions: predicted class or regression value
            probabilities: predicted probabilities
            targets: target value
            mask_type: mask_type: train, test or valid

        Returns: None
        """
        # We find the proper section name
        section = Recorder.MASK_TO_SECTION[RESULTS][mask_type]

        # We save the predictions
        targets = targets if targets is not None else zeros(predictions.shape[0])
        if len(predictions.shape) == 0:
            for j, id_ in enumerate(ids):
                self._data[section][str(id_)] = {
                    PROBABILITY: str(predictions[j].item()),
                    PREDICTION: str(predictions[j].item()),
                    TARGET: str(targets[j].item())}
        else:
            for j, id_ in enumerate(ids):
                self._data[section][str(id_)] = {
                    PROBABILITY: str(probabilities[j].tolist()),
                    PREDICTION: str(predictions[j].tolist()),
                    TARGET: str(targets[j].item())}

    def record_test_predictions(self,
                                ids: List[str],
                                predictions: tensor,
                                targets: tensor) -> None:
        """
        Records the test set's predictions

        Args:
            ids: list of patient/participant ids
            predictions: tensor with predicted targets
            targets: tensor with ground truth

        Returns: None
        """
        return self.record_predictions(ids, predictions, targets, mask_type=MaskType.TEST)

    def record_train_predictions(self,
                                 ids: List[str],
                                 predictions: tensor,
                                 targets: tensor) -> None:
        """
        Records the evaluating set's predictions

        Args:
            ids: list of patient/participant ids
            predictions: tensor with predicted targets
            targets: tensor with ground truth

        Returns: None
        """
        return self.record_predictions(ids, predictions, targets)

    def record_valid_predictions(self,
                                 ids: List[str],
                                 predictions: tensor,
                                 targets: tensor) -> None:
        """
        Records the validation set's predictions

        Args:
            ids: list of patient/participant ids
            predictions: tensor with predicted targets
            targets: tensor with ground truth

        Returns: None
        """
        return self.record_predictions(ids, predictions, targets, mask_type=MaskType.VALID)


def get_evaluation_recap(evaluation_name: str,
                         recordings_path: str,
                         masks_types: List[str] = None) -> None:
    """
    Creates a file with a summary of results from records.json file of each data split

    Args:
        evaluation_name: name of the evaluation
        recordings_path: directory where containing the folders with the results of each split

    Returns: None
    """

    # We check if the directory with results exists
    path = os.path.join(recordings_path, evaluation_name)
    if not os.path.exists(path):
        raise ValueError('Impossible to find the given directory')

    # We sort the folders in the directory according to the split number
    folders = next(os.walk(path))[1]
    folders.sort(key=lambda x: int(x.split("_")[1]))

    # Initialization of an empty dictionary to store the summary
    data = {HYPERPARAMETER_IMPORTANCE: {},
            HYPERPARAMETERS: {},
            COEFFICIENT: {}
            }
    if masks_types is None or len(masks_types) < 1:
        data[TRAIN_METRICS] = {}
        data[TEST_METRICS] = {}

    else:
        for mask_type in masks_types:
            if mask_type is not MaskType.INNER:
                data[mask_type + '_' + METRICS] = {}

    # Initialization of a list of key list that we can found within section of records dictionary
    key_lists = {}

    for folder in folders:

        # We open the json file containing the info of each split
        with open(os.path.join(path, folder, RECORDS_FILE), "r") as read_file:
            split_data = json.load(read_file)

        # For each section and their respective key list
        for section in data.keys():
            if section in split_data.keys():

                # If the key list is not initialized yet..
                if key_lists.get(section) is None:

                    # Initialization of the key list
                    key_lists[section] = split_data[section].keys()

                    # Initialization of each individual key section in the dictionary
                    for key in key_lists[section]:
                        data[section][key] = {VALUES: [], INFO: ""}

                # We add values to each key associated to the current section
                for key in key_lists[section]:
                    data[section][key][VALUES].append(split_data[section][key])

    # We add the info about the mean, the standard deviation, the median , the min, and the max
    set_info(data)

    # We save the json containing the summary of the records
    with open(os.path.join(path, SUMMARY_FILE), "w") as file:
        json.dump(data, file, indent=True)


def set_info(data: Dict[str, Dict[str, Union[List[Union[str, float]], str]]]) -> None:
    """
    Adds the mean, the standard deviation, the median, the min and the max
    to the numerical parameters of each section of the dictionary with the summary.

    Otherwise, counts the number of appearances of the categorical parameters.

    Args:
        data: dictionary with the summary of results from the splits' records

    Returns: None

    """
    # For each section
    for section in data.keys():

        # For each key of this section
        for key in data[section].keys():

            # We extract the list of values
            values = data[section][key][VALUES]

            if values[0] is None:
                data[section][key][INFO] = str(None)
            elif (not isinstance(values[0], str)) & (not isinstance(values[0], bool)):
                mean_, std_ = round(np.mean(values), 4), round(std(values), 4)
                med_, min_, max_ = round(median(values), 4), round(min(values), 4), round(max(values), 4)
                data[section][key][INFO] = f"{mean_} +- {std_} [{med_}; {min_}-{max_}]"
                data[section][key][MEAN] = mean_
                data[section][key][STD] = std_
            else:
                counts = Counter(data[section][key][VALUES])
                data[section][key][INFO] = str(dict(counts))



def plot_hps_importance_chart(evaluation_name: str,
                              recordings_path: str) -> None:
    """
    Creates a bar plot containing information about the mean and standard deviation
    of each hyperparameter's importance.

    Args:
        evaluation_name: name of the evaluation
        recordings_path: directory where containing the folders with the results of each split

    Returns: None

    """
    # We get the content of the json file
    path = os.path.join(recordings_path, evaluation_name)
    with open(os.path.join(path, SUMMARY_FILE), "r") as read_file:
        data = json.load(read_file)[HYPERPARAMETER_IMPORTANCE]

    # We initialize three lists for the values, the errors, and the labels
    values, errors, labels = [], [], []

    # We collect the data of each hyperparameter importance
    for key in data.keys():
        values.append(data[key][MEAN])
        errors.append(data[key][STD])
        labels.append(key)

    # We sort the list according values
    sorted_values = sorted(values)
    sorted_labels = sorted(labels, key=lambda x: values[labels.index(x)])
    sorted_errors = sorted(errors, key=lambda x: values[errors.index(x)])

    # We build the plot
    x_pos = arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.bar(x_pos, sorted_values, yerr=sorted_errors, capsize=5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sorted_labels)
    ax.set_title('Hyperparameters importance')

    # We save the plot
    plt.savefig(os.path.join(path, HPS_IMPORTANCE_CHART))
    plt.close()


def compare_prediction_recordings(evaluations: List[str],
                                  split_index: int,
                                  recording_path: str) -> None:
    """
    Creates a scatter plot showing the predictions of one or two
    experiments against the real labels.

    Args:
        evaluations: list of str representing the names of the evaluations to compare
        split_index: index of the split we want to compare
        recording_path: directory that stores the evaluations folder

    Returns: None
    """

    # We check that the number of evaluations provided is 2
    if not (1 <= len(evaluations) <= 2):
        raise ValueError("One or two evaluations must be specified")

    # We create the paths to recoding files
    paths = [os.path.join(recording_path, e, f"Split_{split_index}", RECORDS_FILE) for e in evaluations]

    # We get the data from the recordings
    all_data = []  # List of dictionaries
    for path in paths:
        # We read the record file of the first evaluation
        with open(path, "r") as read_file:
            all_data.append(json.load(read_file))

    # We check if the two evaluations are made on the same patients
    comparison_possible = True
    first_experiment_ids = list(all_data[0][TEST_RESULTS].keys())

    for i, data in enumerate(all_data[1:]):

        # We check the length of both predictions list
        if len(data[TEST_RESULTS]) != len(all_data[0][TEST_RESULTS]):
            comparison_possible = False
            break

        # We check ids in both list
        for j, id_ in enumerate(data[TEST_RESULTS].keys()):
            if id_ != first_experiment_ids[j]:
                comparison_possible = False
                break

    if not comparison_possible:
        raise ValueError("Different patients are present in the given evaluations")

    targets, ids, all_predictions = [], [], []

    # We gather the needed data from the recordings
    for i, data in enumerate(all_data):

        # We add an empty list to store predictions
        all_predictions.append([])

        for id_, item in data[TEST_RESULTS].items():

            # If we have not registered ids and targets yet
            if i == 0:
                ids.append(id_)
                targets.append(float(item[TARGET]))

            all_predictions[i].append(float(item[PREDICTION]))

    # We sort predictions and the ids based on their targets
    indexes = list(range(len(targets)))
    indexes.sort(key=lambda x: targets[x])
    all_predictions = [[predictions[i] for i in indexes] for predictions in all_predictions]
    targets = [targets[i] for i in indexes]
    ids = [ids[i] for i in indexes]

    # We set some parameters of the plot
    plt.rcParams["figure.figsize"] = (15, 6)
    plt.rcParams["xtick.labelsize"] = 6

    # We create the scatter plot
    colors = ["blue", "orange"]
    plt.scatter(ids, targets, color="green", label="ground truth")
    for i, predictions in enumerate(all_predictions):
        plt.scatter(ids, predictions, color=colors[i], label=evaluations[i])

    # We add the legend and the title to the plot
    plt.legend()
    plt.title("Predictions and ground truth")

    # We save the plot
    plt.savefig(os.path.join(recording_path, evaluations[0], f"Split_{split_index}",
                             f"comparison_{'_'.join(evaluations)}.png"))
    plt.close()


def load_rf_homr_results(train_set: pd.DataFrame,
                         test_set: pd.DataFrame = None,
                         internal_file_path: str = None,
                         external_file_path: str = None,
                         internal: bool = False,
                         external: bool = False,
                         unique_test_visits: str = None,
                         unique_holdout_visits: str = None) -> None:
    """
    Loads train and test results of HOMR experiment made by Ryeyan Taseen to report metrics

    Args:
        train_set: learning set of HOMR experiment
        test_set: testing set of HOMR experiment
        internal_file_path: file to results of internal validation
        external_file_path: file to results of external experiments

    Returns: None
    """
    # Set evaluation metrics
    evaluation_metrics = [BinaryAccuracy(), BinaryBalancedAccuracy(),
                          BinaryBalancedAccuracy(Reduction.GEO_MEAN),
                          Sensitivity(), Specificity(), AUC(), BrierScore(),
                          Precision(), F1Score(), F2Score()
                          ]

    # Stores all visits ids of evaluating set in an array which indexes are observations ids
    train_visits = np.array(train_set[VISIT], dtype=str)

    if internal:
        # Load RF predictions obtained in internal validation of HOMR experiment
        internal_results = pd.read_csv(internal_file_path)

        evaluation_name = 'RF_HOMR_internal'

        # Get each visit selected for each patient if we want a single visit prediction
        unique_visits = pd.read_csv(unique_test_visits)[
            ['visit_id', 'fold', 'predicted']] if unique_test_visits is not None else None

        # Record results of the internal validation splits
        for k in range(10):
            # Select the evaluating and testing results of the Kth validation split
            k_df = internal_results[internal_results['fold'] == k + 1]

            # Create the recorder of the Kth split
            train_recorder = Recorder(evaluation_name, k, Paths.EXPERIMENTS_RECORDS)
            single_visits = np.array(unique_visits[unique_visits['fold'] == (k + 1)]['visit_id'], dtype=str) if \
                unique_test_visits is not None else None
            single_predictions = np.array(unique_visits[unique_visits['fold'] == (k + 1)]['predicted'], dtype=float) if \
                unique_test_visits is not None else None
            record_test_train_result_homr(k_df, train_set, train_visits, train_recorder, evaluation_metrics,
                                          single_visits=single_visits, predictions=single_predictions)

        # We save the evaluation recap
        get_evaluation_recap(evaluation_name=evaluation_name, recordings_path=Paths.EXPERIMENTS_RECORDS)

    if external:
        # Load RF predictions obtained in external validation of HOMR experiment
        external_results = pd.read_csv(external_file_path)

        # Stores all visits ids of testing set in an array which indexes are observations ids
        test_visits = np.array(test_set[VISIT], dtype=str)

        evaluation_name = 'RF_HOMR_external'

        # Get each visit selected for each patient if we want a single visit prediction
        unique_visits = pd.read_csv(unique_holdout_visits)[
            ['visit_id', 'predicted']] if unique_holdout_visits is not None else None

        single_visits = np.array(unique_visits['visit_id'], dtype=str) if unique_holdout_visits is not None else None
        single_predictions = np.array(unique_visits['predicted'],
                                      dtype=float) if unique_holdout_visits is not None else None

        # Create the recorder
        test_recorder = Recorder(evaluation_name, 0, Paths.EXPERIMENTS_RECORDS)
        record_test_train_result_homr(external_results, train_set, train_visits, test_recorder, evaluation_metrics,
                                      test_set, test_visits, single_visits=single_visits,
                                      predictions=single_predictions)

        # We save the evaluation recap
        get_evaluation_recap(evaluation_name=evaluation_name, recordings_path=Paths.EXPERIMENTS_RECORDS)


def record_test_train_result_homr(results: pd.DataFrame,
                                  train_set: pd.DataFrame,
                                  train_visits: array,
                                  recorder: Recorder,
                                  evaluation_metrics: List[Metric],
                                  test_set: pd.DataFrame = None,
                                  test_visits: array = None,
                                  single_visits: array = None,
                                  predictions: array = None) -> None:
    """
    Records train and test results of HOMR experiment made by Ryeyan Taseen

    Args:
        results: results of the experiment in a pandas dataframe
        train_set: evaluating set of the experiment
        train_visits: visits_ids of the evaluating observations of the experiment
        recorder: object recorder
        evaluation_metrics: metrics we want to report
        test_set: testing set of the experiment
        test_visits: visits_ids of the testing observations of the experiment
        single_visits: array of visits where each patient has only one visit considered

    Returns: None
    """

    for mask_type in [MaskType.TRAIN, MaskType.TEST]:
        # Get the predictions of the appropriate set
        if mask_type is MaskType.TEST and test_set is not None:
            set = test_set
            set_visits = test_visits
        else:
            set = train_set
            set_visits = train_visits

        is_test = 't' if mask_type is MaskType.TEST else 'f'
        dataset = results[results['is_test'] == is_test]

        # Get ids of the observations
        visits = dataset['visit_id'] if ((single_visits is None) | (mask_type is not MaskType.TEST)) else single_visits
        ids = []
        for visit in visits:
            ids.append(np.where(set_visits == visit)[0][0])

        # Get the targets from the dataset
        targets = np.array(set[OYM].values[ids], dtype=int)

        # Get the predictions
        proba = np.array(dataset['predicted'], dtype=float) if \
            ((single_visits is None) | (mask_type is not MaskType.TEST)) else predictions

        str_ids = [str(id) for id in ids]

        # Get the optimal threshold
        if mask_type is MaskType.TRAIN:
            threshold = HOMRBinaryClassifier.optimize_J_statistic(targets, proba)

        record_external_results(str_ids, targets, proba, mask_type, recorder, evaluation_metrics, threshold)


def record_external_results(ids: List[str],
                            targets: np.array,
                            proba: np.array,
                            mask_type: str,
                            recorder: Recorder,
                            evaluation_metrics: List[Metric],
                            threshold: float = 0.5) -> None:
    """
    record results of an external experiment in a JSON file

       Args:
           ids: observation ids
           targets: ground truth of each observation
           proba: probabilities of each observation to be in class 1
           mask_type: train, valid or test
           recorder: object recorder
           evaluation_metrics: metrics to report about he experiement

       Returns: None
    """

    # We record all metric scores
    for metric in evaluation_metrics:
        recorder.record_scores(score=metric(proba, targets, thresh=threshold),
                               metric=metric.name, mask_type=mask_type)

    # Get the final predictions using hte optimal threshold
    pred = (from_numpy(proba) >= threshold).long()

    # We save the predictions
    recorder.record_predictions(predictions=pred, ids=ids, targets=targets, mask_type=mask_type)

    # We save all the data collected in one file
    recorder.generate_file()


def record_probabilities(dataset: HOMRDataset,
                         model: BinaryClassifier,
                         path: str,
                         file: str = None) -> Optional[Dict]:
    """
    Records probabilities (specific for XGBclassifier right now)

       Args:
           dataset: HOMR dataset
           model: HOMR Binary Classifier model
           path: path to the directory where the predictions will be stored
           file: name of the file where the predictions will be stored

    Returns: None
    """

    # Get predicted proba for each
    probabilities = model.predict_proba(dataset.x)

    # Build the dictionary mapping each participant ID to its predicted probability
    dictionary = {}
    for _id in dataset.ids:
        dictionary[str(_id)] = str(probabilities[_id, 1])

    if file is not None:
        # Serialize json object
        json_object = json.dumps(dictionary)

        # Open json file and save predictions
        with open(os.path.join(path, file + '.json'), 'w') as saving_file:
            saving_file.write(json_object)

    else:
        return dictionary


def load_model(path_directory: str = 'records/experiments/XGBoost_dxadm_dep10/Split_1',
               records_file: str = 'records.json',
               model_file: str = 'sklearn_model.sav'
               ):
    # load the model
    mf = open(os.path.join(path_directory, model_file), 'rb')
    model = pickle.load(mf)

    # load the records file
    rf = open(os.path.join(path_directory, records_file))
    records = json.load(rf)

    threshold = records['data_info']["thresh"]

    return model, threshold


def record_predictions(dataset: HOMRDataset,
                       model: BinaryClassifier,
                       path: str,
                       file: str = None,
                       threshold: float = 0.5):
    """
        Records predictions (specific for XGBclassifier right now)

           Args:
               dataset: HOMR dataset
               model: HOMR Binary Classifier model
               path: path to the directory where the predictions will be stored
               file: name of the file where the predictions will be stored

        Returns: None
    """

    # Get predicted proba for each
    predictions = model.predict_proba(dataset.x)

    # Build the dictionary mapping each participant ID to its predicted probability
    dictionary = {}
    for _id in dataset.ids:
        dictionary[str(_id)] = str(int(predictions[_id, 1] > float(threshold)))

    if file is not None:
        # Serialize json object
        json_object = json.dumps(dictionary)

        # Open json file and save predictions
        with open(os.path.join(path, file + '.json'), 'w') as saving_file:
            saving_file.write(json_object)

    else:
        return dictionary


def record_precision(experiment_file: str,
                     n_splits: int):
    precision = {}
    positive_label = {}

    for i in range(n_splits):
        f = open(os.path.join(experiment_file, 'Split_' + str(i) + '/records.json'))
        records = json.load(f)

        for mask in [MaskType.TRAIN, MaskType.VALID, MaskType.TEST]:
            targets, predictions = [], []
            # Get the targets and predictions from the json file
            for _, pred_tar in records[mask + '_results'].items():
                targets.append(int(pred_tar['target']))
                predictions.append(int(pred_tar['prediction']))

            # Get the precision from the confusion matrix
            conf_mat = np.zeros((2, 2))
            for t, p in zip(targets, predictions):
                conf_mat[t, p] += 1
            positive_label[mask] = (conf_mat[1, 1] + conf_mat[0, 1])
            try:
                precision[mask].append((conf_mat[1, 1] / (conf_mat[1, 1] + conf_mat[0, 1])).item())
            except:
                precision[mask] = []
                precision[mask].append(round((conf_mat[1, 1] / (conf_mat[1, 1] + conf_mat[0, 1])).item(), 4))

            records[mask + '_metrics']['Precision'] = precision[mask][i]
            records[mask + '_metrics']['PositiveLabels'] = positive_label[mask]
        # Save the precision metric in the json file
        with open(os.path.join(experiment_file, 'Split_' + str(i) + '/records.json'), 'w') as records_precision:
            json.dump(records, records_precision, indent=True)
            records_precision.close()

    # Read summary file
    f_summary = open(os.path.join(experiment_file, 'summary.json'))
    summary = json.load(f_summary)

    # Record in the summary file
    for mask in [MaskType.TRAIN, MaskType.TEST]:
        values = np.array(precision[mask])
        section = mask + '_metrics'
        key = 'Precision'
        summary[section][key] = {}
        mean_, std_ = round(np.mean(values), 4), round(std(values), 4)
        med_, min_, max_ = round(median(values), 4), round(min(values), 4), round(max(values), 4)
        summary[section][key]['info'] = f"{mean_} +- {std_} [{med_}; {min_}-{max_}]"
        summary[section][key]['mean'] = mean_
        summary[section][key]['std'] = std_

    with open(os.path.join(experiment_file, 'summary.json'), 'w') as summary_precision:
        json.dump(summary, summary_precision, indent=True)
        summary_precision.close()


def record_map_probabilities_idx(probabilities_visits_file: str,
                                 dataset: pd.DataFrame,
                                 saving_file: str,
                                 name_file: str):
    probabilities_visits = pd.read_csv(probabilities_visits_file)
    ids = np.sort(array(dataset[IDS]))
    ids_probabilities = []
    print(len(ids))
    print(len(np.unique(ids)))

    # find the
    for _id in ids:
        print(_id)
        # find the visit id of the corresponding hospitalisation
        visit_id = dataset[VISIT].loc[_id]

        # find the probability predicted by HOMR model
        probability = probabilities_visits[probabilities_visits[VISIT] == visit_id]['predicted']

        # Store the prediction in the list
        ids_probabilities.append(probability)

    ids_probabilities = array(ids_probabilities)

    # sort the ids in the dictionary and build the dataframe
    # dataset = pd.DataFrame.from_dict({key: ids_probabilities[key] for key in sorted(ids_probabilities.keys())})
    # Save the dataset
    with open(os.path.join(saving_file, name_file), 'wb') as saving_file:
        print(saving_file)
        pickle.dump(ids_probabilities, saving_file)


def records_homr_metrics_different_masks(masks: Dict,
                                         probabilities: array,
                                         targets: array,
                                         n_positive_values: int):
    evaluation_metrics = [BinaryAccuracy(),
                          BinaryBalancedAccuracy(Reduction.GEO_MEAN),
                          Sensitivity(), Specificity(), AUC(), BrierScore(),
                          BalancedAccuracyEntropyRatio(Reduction.GEO_MEAN),
                          Precision(), NegativePredictiveValue(),
                          NFN(), NFP(), NTN(), NTP(),
                          F2Score(), F1Score(), BinaryBalancedAccuracy()]

    for i, (k, v) in enumerate(masks.items()):
        # We create the Recorder object to save the result of this experience
        # We extract the masks
        train_mask, valid_mask = v[MaskType.TRAIN], v[MaskType.VALID]
        test_mask, in_masks = v[MaskType.TEST], v[MaskType.INNER]

        # Extract other test masks
        additional_test_masks = {}
        for (mask_type, idx_mask) in v.items():
            if mask_type not in [MaskType.TRAIN, MaskType.VALID, MaskType.TEST, MaskType.INNER]:
                additional_test_masks[mask_type] = idx_mask

        recorder = Recorder(evaluation_name='RF_HOMR_external_all_masks',
                            index=0,
                            recordings_path=Paths.EXPERIMENTS_RECORDS,
                            masks_types=list(v.keys()))

        for name, mask in [("train_set", train_mask), ("valid_set", valid_mask), ("test_set", test_mask)]:
            mask_length = len(mask) if mask is not None else 0
            recorder.record_data_info(name, mask_length)

        # Record the data count for the additional test masks if there is any
        for name, mask in additional_test_masks.items():
            mask_length = len(mask) if mask is not None else 0
            recorder.record_data_info('test_' + name + '_set', mask_length)

        m = [tuple(zip(additional_test_masks.values(), additional_test_masks.keys()))[i] for i in
             range(len(additional_test_masks.values()))]

        mask_list = [(train_mask, MaskType.TRAIN),
                     (valid_mask, MaskType.VALID),
                     (test_mask, MaskType.TEST)] + m

        for mask, mask_type in mask_list:
            if len(mask) > 0:
                pred = probabilities[list(mask)]
                pred = np.squeeze(pred, 1)
                ids = mask
                y = targets[list(mask)]

                # FInd optimal threshold
                if mask_type is MaskType.TRAIN:
                    thresh = HOMRBinaryClassifier.optimize_J_statistic(y, pred)
                    recorder.record_data_info('thresh', str(thresh))

                if (n_positive_values is not None) and (mask_type == 'fixed_alerts'):
                    threshold = min(sorted(pred, reverse=True)[:n_positive_values])
                    recorder.record_data_info('positive_thresh', str(threshold))

                else:
                    threshold = thresh

                # We record all metric scores
                for metric in evaluation_metrics:
                    recorder.record_scores(score=metric(pred, y, thresh=threshold),
                                           metric=metric.name, mask_type=mask_type)

                if not is_tensor(pred):
                    pred = from_numpy(pred)

                # We get the final predictions from the soft predictions
                pred = (pred >= threshold).long()

                # We save the predictions
                recorder.record_predictions(predictions=pred, ids=ids, targets=y, mask_type=mask_type)

                # We save all the data collected in one file
                recorder.generate_file()
