"""
Filename: delong.py

Description: This file is used to implement the DeLong test,
            code taken from : https://github.com/yandexdataschool/roc_comparison

"""
import json
import os
from re import search

import numpy as np
import scipy.stats


# AUC comparison adapted from
# https://github.com/Netflix/vmaf/
def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Operating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float)
    ty = np.empty([k, n], dtype=np.float)
    tz = np.empty([k, m + n], dtype=np.float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    """Computes log(10) of p-values.
    Args:
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
    Returns:
       log10(pvalue)
    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)


def compute_ground_truth_statistics(ground_truth):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    return order, label_1_count


def delong_roc_variance(ground_truth, predictions):
    """
    Computes ROC AUC variance for a single set of predictions
    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov


def delong_roc_test(ground_truth, predictions_one, predictions_two):
    """
    Computes log(p-value) for hypothesis that two ROC AUCs are different
    Args:
       ground_truth: np.array of 0 and 1
       predictions_one: predictions of the first model,
          np.array of floats of the probability of being class 1
       predictions_two: predictions of the second model,
          np.array of floats of the probability of being class 1
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    return calc_pvalue(aucs, delongcov), aucs

def compute_median_significant_difference(best_model: str,
                                          second_model: str,
                                          n_splits: int,
                                          saving_dir: str,
                                          section_best_model: str = None,
                                          section_second_model: str = None,
                                          temporal_analysis: bool = False,
                                          sec: str = '/records.json'
                                          ):
    # Validation of inputs
    if temporal_analysis & ((section_second_model is None) | (section_best_model is None)):
        raise ValueError("section must be specified for temporal analysis")

    # Create saving directory
    os.makedirs(saving_dir)
    p_values = {}

    # Iterate over all the splits
    for i in range(n_splits):
        # Read the data recorded at each split for both models
        with open(os.path.join(best_model, 'Split_' + str(i) + sec), "r") as file:
            best_model_data = json.load(file)

        with open(os.path.join(second_model, 'Split_' + str(i) + sec), "r") as file:
            second_model_data = json.load(file)

        if temporal_analysis | (section_second_model is None) | (section_best_model is None):

            for section in best_model_data.keys():

                if search("(test).*(results)$", section):
                    if section not in p_values.keys():
                        p_values[section] = {}
                    section_a = section
                    section_b = section
                    if (section_second_model is not None) & (section_best_model is not None):
                        if section == 'test_results':
                            section_a = section
                            section_b = section_second_model
                        elif section == section_best_model:
                            section_a = section
                            section_b = 'test_results'

                    best_model_probabilities = []
                    second_model_probabilities = []
                    targets = []

                    # Get all the probabilities predicted in the current test set and the original targets
                    for _id in best_model_data[section_a].keys():
                        best_model_probabilities.append(float(best_model_data[section_a][_id]['probability']))
                        second_model_probabilities.append(float(second_model_data[section_b][_id]['probability']))

                        best_model_target = int(best_model_data[section_a][_id]['target'])
                        second_model_target = int(second_model_data[section_b][_id]['target'])

                        if best_model_target != second_model_target:
                            print('error, different targets')
                        else:
                            targets.append(best_model_target)

                    # Compute the p-value with delong test that return long10(p-value)
                    p_value, auc = delong_roc_test(np.array(targets),
                                                   np.array(best_model_probabilities),
                                                   np.array(second_model_probabilities))
                    p_val = 10 ** p_value[0][0]

                    if (p_val <= 0.1) & (auc[0] < auc[1]):
                        p_val = p_val + 1

                    p_values[section][str(i)] = {}
                    p_values[section][str(i)]['p_value'] = str(p_val)
                    p_values[section][str(i)]['auc'] = str(auc)

        else:
            best_model_probabilities = []
            second_model_probabilities = []
            targets = []
            for _id in best_model_data[section_best_model].keys():
                best_model_probabilities.append(float(best_model_data[section_best_model][_id]['probability']))
                second_model_probabilities.append(float(second_model_data[section_second_model][_id]['probability']))

                best_model_target = int(best_model_data[section_best_model][_id]['target'])
                second_model_target = int(second_model_data[section_second_model][_id]['target'])

                if best_model_target != second_model_target:
                    print('error, different targets')
                else:
                    targets.append(best_model_target)

            # Compute the p-value with delong test that return long10(p-value)
            p_value = 10 ** delong_roc_test(np.array(targets),
                                            np.array(best_model_probabilities),
                                            np.array(second_model_probabilities))[0][0]

            p_values[str(i)] = str(p_value)

    # Combine p-values with Fisher's method: https://en.wikipedia.org/wiki/Fisher%27s_method
    if (section_second_model is not None) & (section_best_model is not None) & (not temporal_analysis):
        p_values['median'] = np.median(np.array([float(p_value) for p_value in p_values.values()]))
        p_values['min'] = np.min(np.array([float(p_value) for p_value in p_values.values()]))
        p_values['max'] = np.max(np.array([float(p_value) for p_value in p_values.values()]))
    else:
        for section, p_val_section in p_values.items():
            p_values[section]['median'] = np.median(
                np.array([float(p_val_section[str(m)]['p_value']) for m in range(n_splits)]))
            p_values[section]['min'] = np.min(
                np.array([float(p_val_section[str(m)]['p_value']) for m in range(n_splits)]))
            p_values[section]['max'] = np.max(
                np.array([float(p_val_section[str(m)]['p_value']) for m in range(n_splits)]))

    with open(os.path.join(saving_dir, 'p_values.json'), "w") as file:
        json.dump(p_values, file, indent=True)