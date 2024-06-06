"""
Filename: sampling.py

Authors: Hakima Laribi
         Nicolas Raymond

Description: Defines the classes used to sample the dataset for analyses

"""
from collections import OrderedDict
from random import shuffle
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
from numpy import array
from numpy.random import seed
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from torch.utils.data import Sampler
from tqdm import tqdm

from src.data.processing import constants
from src.data.processing.datasets import HOMRDataset


class MaskType:
    """
    Stores the constant related to mask types
    """
    TRAIN: str = "train"
    VALID: str = "valid"
    TEST: str = "test"
    INNER: str = "inner"

    def __iter__(self):
        return iter([self.TRAIN, self.VALID, self.TEST])


class OneShotSampler:
    """
    Object used in order to retrieve one single list of indexes to use as train and test masks in the final evaluation,
    training and testing sets are priorly given
    """

    def __init__(self,
                 dataset: HOMRDataset,
                 random_state: int = 101,
                 valid_size: float = 0.20,
                 n_inner: int = 0
                 ):
        """
        Sets protected attributes of the sampler

        Args:
            dataset: HOMR dataset
            random_state: random state to reproduce results
            valid_size: percentage of training data to be used for validation
            n_inner: number of folds in inner splits
        """

        # Sets protected attributes
        self._dataset = dataset
        self._random_state = random_state
        self._valid_size = valid_size
        self._inner = n_inner

    def __call__(self,
                 learning_ids,
                 test_ids,
                 sampling_strategy: int = None,
                 multiple_test_masks: bool = True,
                 serialize: bool = False,
                 ) -> Dict:
        """
        Returns lists of indexes to use as train and test masks for outer and inner splits

        Args:
            learning_ids: list of ids of patients from the learning set
            test_ids: list of ids of patients from the holdout set
            sampling_strategy: samples to consider in the training and validation sets
                            - if i > 0, the ith id's occurrence to consider (ith patient's visit)
                            - if -1 consider the last occurrence (visit).
                            - if None, consider all patient's visits' independently
            multiple_test_masks: boolean to specify if we build multiple test masks in the outer loop.
            serialize: boolean to specify whether to organize the list of ids in temporal sequences for RNN inputs
        Example:
            {0: {'train': [..], 'test': [..]}, 'inner': [..]}

        """
        masks = {}

        # Get the data samples and target variables
        map_ids = self._dataset.map_ids_to_indexes()

        # Set random for reproducibility
        seed(self._random_state)

        if self._valid_size > 0.:
            train_ids, valid_ids = train_test_split(learning_ids, test_size=self._valid_size, random_state=1101,
                                                    shuffle=True)
        else:
            train_ids = learning_ids
            valid_ids = None

        # Get the corresponding indexes in the dataset of the selected ids according to the sampling strategy
        list_test_ids = test_ids if isinstance(test_ids[0], list) else [test_ids]
        for i in range(len(list_test_ids)):
            test_ids = list_test_ids[i]
            masks[i] = KFoldsSampler.sample_idx_with_strategy(multiple_test_masks,
                                                              map_ids,
                                                              (train_ids, valid_ids, test_ids),
                                                              sampling_strategy,
                                                              serialize
                                                              )
            masks[i][MaskType.INNER] = {}

            if self._inner > 0:
                inner_k_folds = KFold(n_splits=self._inner, random_state=self._random_state, shuffle=True)

                inner_splits = inner_k_folds.split(train_ids)
                with tqdm(total=self._inner) as bar:
                    for j, (inner_remaining_idx, inner_test_idx) in enumerate(inner_splits):
                        # Get the ids selected in each set (not the indexes)
                        inner_test_ids = np.array(train_ids)[inner_test_idx]
                        inner_learning_ids = np.array(train_ids)[inner_remaining_idx]

                        # Get the training and validation masks
                        if self._valid_size > 0.:
                            inner_train_ids, inner_valid_ids = train_test_split(inner_learning_ids,
                                                                                test_size=self._valid_size,
                                                                                random_state=101,
                                                                                shuffle=True)
                        else:
                            inner_train_ids = inner_learning_ids
                            inner_valid_ids = None

                        # Get the corresponding indexes in the dataset of the selected ids according to the sampling
                        # strategy
                        masks[i][MaskType.INNER][j] = KFoldsSampler.sample_idx_with_strategy(False,
                                                                                             map_ids,
                                                                                             (inner_train_ids,
                                                                                              inner_valid_ids,
                                                                                              inner_test_ids),
                                                                                             sampling_strategy,
                                                                                             serialize)
                        bar.update()

        return masks


class KFoldsSampler:
    """
    Object used in order to generate lists of indexes to use as train and test masks, each fold
    is used once as a test set while the k - 1 remaining folds form the training set, a sample is chosen
    only once in the K test folds.
    """

    def __init__(self,
                 dataset: HOMRDataset,
                 k: int = 10,
                 inner_k: int = 10,
                 random_state: int = 101,
                 valid_size: float = 0.20,
                 folds: Optional[array] = None
                 ):
        """
        Sets protected attributes of the sampler

            Args:
                dataset: HOMR dataset
                k: number of folds in outer splits
                inner_k : number of folds in inner splits
                random_state: random state to reproduce results
                valid_size: percentage of training data to be used for validation
                folds: (N,) array with the same size of the dataset containing the fold of each sample
        """
        # Validation of input
        if k <= 0:
            raise ValueError('Number of inner and outer folds must be greater than 0')
        if folds is not None and len(folds) != len(dataset):
            raise ValueError('Length of folds array must equal length of the dataset')
        if folds is not None and len(np.unique(folds)) != k:
            raise ValueError(f'Number of folds in folds array = {len(np.unique(folds))} different from k = {k}')

        # Sets protected attributes
        self._dataset = dataset
        self._k = k
        self._inner_k = inner_k
        self._random_state = random_state
        self._folds = folds
        self._valid_size = valid_size

    def __call__(self,
                 sampling_strategy: int = None,
                 multiple_test_masks: bool = True,
                 experiment: Optional[str] = None,
                 serialize: bool = False
                 ) -> Dict:
        """
        Returns lists of indexes to use as train and test masks for outer and inner folds cross validation splits

        Args:
            sampling_strategy: samples to consider in the training and validation sets
                            - if i > 0, the ith id's occurrence to consider (ith patient's visit)
                            - if -1 consider the last occurrence (visit).
                            - if None, consider all patient's visits' independently
            multiple_test_masks: boolean to specify if we build multiple test masks in the outer loop.
            serialize: boolean to specify whether to organize the list of ids in temporal sequences for RNN inputs

        Example:
            {0: {'train': [..], 'test': [..]}, 1:{'train': [..], 'test': [..]}, ..., k:{'train': [..], 'test': [..]}}

        """
        masks = {}

        # Get the data samples and target variables
        map_ids = self._dataset.map_ids_to_indexes()
        ids = [key for key in map_ids.keys()]
        y = self._dataset.y

        # If outer folds are not specified
        if self._folds is None:
            # Get the K outer splits of K fold cross validation
            k_folds = KFold(n_splits=self._k, random_state=self._random_state, shuffle=True)
            splits = k_folds.split(ids)

        # If the outer splits are given
        else:
            # Get the K outer splits of K fold cross validation
            splits = [(self._folds != fold, self._folds == fold) for fold in np.unique(self._folds)]

        # Get the inner_k inner splits of K fold cross validation
        if self._inner_k > 0:
            inner_k_folds = KFold(n_splits=self._inner_k, random_state=self._random_state, shuffle=True)

        # Set random for reproducibility
        seed(self._random_state)

        # Get the outer and inner masks
        with tqdm(total=(self._k + self._k * self._inner_k)) as bar:
            # Get the K sets of training, testing and inner masks
            for i, (remaining_idx, test_idx) in enumerate(splits):
                # Get the ids selected in each set (not the indexes)
                test_ids = np.array(ids)[test_idx]
                learning_ids = np.array(ids)[remaining_idx]

                # Get the training and validation masks
                if self._valid_size > 0.:
                    train_ids, valid_ids = train_test_split(learning_ids, test_size=self._valid_size,
                                                            random_state=self._random_state,
                                                            shuffle=True)
                else:
                    train_ids = learning_ids
                    valid_ids = None

                # Get the corresponding indexes in the dataset of the selected ids according to the sampling strategy
                masks[i] = self.sample_idx_with_strategy(multiple_test_masks,
                                                         map_ids,
                                                         (train_ids, valid_ids, test_ids),
                                                         sampling_strategy,
                                                         serialize)
                masks[i][MaskType.INNER] = {}
                bar.update()

                if self._inner_k > 0:
                    inner_splits = inner_k_folds.split(train_ids)
                    for j, (inner_remaining_idx, inner_test_idx) in enumerate(inner_splits):
                        # Get the ids selected in each set (not the indexes)
                        inner_test_ids = np.array(train_ids)[inner_test_idx]
                        inner_learning_ids = np.array(train_ids)[inner_remaining_idx]

                        # Get the training and validation masks
                        if self._valid_size > 0.:
                            inner_train_ids, inner_valid_ids = train_test_split(inner_learning_ids,
                                                                                test_size=self._valid_size,
                                                                                random_state=self._random_state,
                                                                                shuffle=True)
                        else:
                            inner_train_ids = inner_learning_ids
                            inner_valid_ids = None

                        # Get the corresponding indexes in the dataset of the selected ids according to the sampling
                        # strategy
                        masks[i][MaskType.INNER][j] = self.sample_idx_with_strategy(False,
                                                                                    map_ids,
                                                                                    (inner_train_ids,
                                                                                     inner_valid_ids,
                                                                                     inner_test_ids),
                                                                                    sampling_strategy,
                                                                                    serialize
                                                                                    )
                        bar.update()

        return masks

    @staticmethod
    def sample_idx_with_strategy(multiple_test_masks: bool,
                                 map_idx: Dict[int, Dict[int, int]],
                                 ids_masks: Tuple[List[int], Union[List[int], None], List[int]],
                                 sampling_strategy: int = None,
                                 serialize: bool = False):
        """
        Returns lists of indexes to use as train, valid and test masks for one single split, following
        a sampling strategy

        Args:
            multiple_test_masks: boolean to specify if we build multiple test masks.
            map_idx: dictionary mapping each patient id to the corresponding visit occurrences and their indexes.
                        Example: {'patient_1':{'visit_1':index_i, .. ,'visit_k':index_j}}
            ids_masks: tuple with patient ids belonging to the training, validation and testing sets, respectively.
            sampling_strategy: samples to consider in the training and validation sets
                            - if i > 0, the ith id's occurrence to consider (ith patient's visit)
                            - if -1 consider the last occurrence (visit).
                            - if None, consider all patient's visits' independently
            serialize: boolean to specify whether to organize the list of ids in temporal sequences for RNN inputs

        """
        idx_masks = {MaskType.TRAIN: [],
                     MaskType.VALID: [],
                     MaskType.TEST: []}

        temporal_sampling = True if sampling_strategy is not None else False
        max_len = [len(i) for i in map_idx.values()]
        maximum = max(max_len) if ((sampling_strategy == -1) | (sampling_strategy is None)) else sampling_strategy

        for ids, mask_type in zip(ids_masks, idx_masks.keys()):
            if ids is not None:
                # If we consider patients visits independently
                if not temporal_sampling:
                    # Get the last visits for each set: train, valid, set
                    idx_masks[mask_type] += [map_idx[_id][max(map_idx[_id].keys())] for _id in ids]

                    if mask_type != MaskType.TEST:
                        # Add all the visits before the last visit in train and valid sets
                        idx_masks[mask_type] += [map_idx[_id][k] for _id in ids
                                                 for k in map_idx[_id].keys()
                                                 if k < max(map_idx[_id].keys())]
                    else:
                        # Select one random per patient in the testing set
                        idx_masks[MaskType.TEST + '_random_visit'] = [
                            map_idx[_id][np.random.choice(list(map_idx[_id].keys()))]
                            for _id in ids
                        ]

                # If we build sequences of visits for each patient (temporal analysis)
                else:
                    # Get the sequences of visits for train and valid sets according to the sampling strategy
                    if mask_type != MaskType.TEST:
                        idx_masks[mask_type] = [KFoldsSampler.extract_occurrences_idx(maximum, map_idx[_id])
                                                for _id in ids]
                    else:
                        # Get the visit sequence up to the last visit for each patient in the testing set
                        idx_masks[mask_type] = [
                            KFoldsSampler.extract_occurrences_idx(len(map_idx[_id].keys()), map_idx[_id])
                            for _id in ids]
                        # Get the visit sequence up to a randomly selected visit for each patient in the testing set
                        idx_masks[mask_type + '_random_visit'] = [
                            KFoldsSampler.extract_occurrences_idx(
                                np.random.choice(np.arange(1, len(map_idx[_id].keys()) + 1)), map_idx[_id])
                            for _id in ids]

                # Create masks for each group of patients
                if (mask_type == MaskType.TEST) & multiple_test_masks:
                    # Get all the visits of patients with exactly i visits or with at least i visits
                    for i in range(1, constants.MAX_VISIT + 1):
                        section_exact = mask_type + f"_exact_{i}_visits"
                        section_at_least = mask_type + f"_with_{i}_visits"
                        if not temporal_sampling:
                            idx_masks[section_exact] = [map_idx[_id][max(map_idx[_id].keys())]
                                                        for _id in ids
                                                        if len(map_idx[_id].keys()) == i]

                            idx_masks[section_at_least] = [map_idx[_id][sorted(map_idx[_id].keys())[i - 1]]
                                                           for _id in ids
                                                           if len(map_idx[_id].keys()) >= i]

                        else:
                            idx_masks[section_exact] = [
                                KFoldsSampler.extract_occurrences_idx(i, map_idx[_id])
                                for _id in ids
                                if len(map_idx[_id].keys()) == i]
                            idx_masks[section_at_least] = [
                                KFoldsSampler.extract_occurrences_idx(i, map_idx[_id])
                                for _id in ids
                                if len(map_idx[_id].keys()) >= i]

                    # Get the remaining patients having more than MAX_VISIT visits
                    if not temporal_sampling:
                        idx_masks[mask_type + '_other_visit'] = [map_idx[_id][max(map_idx[_id].keys())]
                                                                 for _id in ids
                                                                 if len(map_idx[_id].keys()) > constants.MAX_VISIT
                                                                 ]
                        idx_masks[mask_type + '_other_random_visit'] = [
                            map_idx[_id][
                                sorted(map_idx[_id].keys())[np.random.choice(
                                    np.arange(constants.MAX_VISIT + 1, len(map_idx[_id].keys()) + 1)) - 1]]
                            for _id in ids
                            if len(map_idx[_id].keys()) > constants.MAX_VISIT
                        ]


                    else:
                        idx_masks[mask_type + '_other_visit'] = [
                            KFoldsSampler.extract_occurrences_idx(len(map_idx[_id].keys()), map_idx[_id])
                            for _id in ids
                            if len(map_idx[_id].keys()) > constants.MAX_VISIT]

                        idx_masks[mask_type + '_other_random_visit'] = [
                            KFoldsSampler.extract_occurrences_idx(
                                np.random.choice(np.arange(constants.MAX_VISIT + 1, len(map_idx[_id].keys()) + 1)),
                                map_idx[_id])
                            for _id in ids
                            if len(map_idx[_id].keys()) > constants.MAX_VISIT
                        ]

        if serialize:
            return KFoldsSampler.serialize_mask(idx_masks)
        else:
            return idx_masks

    @staticmethod
    def extract_occurrences_idx(max_occurrence: int,
                                map_idx: Dict[int, int]) -> List[int]:
        """
        Extract the indexes of all the visits before the max_occurrence visit

        Args:
            max_occurrence: the maximum number of visits to extract for each patient
            map_idx: dictionary mapping each patient id to the corresponding visit occurrences and their indexes.
                        Example: {'patient_1':{'visit_1':index_i, .. ,'visit_k':index_j}}

        """
        # Order the occurrences from the first one to the last
        ordered_occ = dict(sorted(map_idx.items()))

        # Get the indexes of all the occurrences <= max_occurrence
        return [ordered_occ[occurrence] for nb, occurrence in enumerate(ordered_occ.keys())
                if nb + 1 <= max_occurrence]

    @staticmethod
    def serialize_mask(mask: Dict[str, List[int]]) -> dict[str, List[List[int]]]:
        """
        Transform the list of indexes to a list of lists of indexes for RNN processing

        Args:
            mask: indexes belonging to training, validation and other test sets of one single split

        """
        serialized_mask = {}
        for key, values in mask.items():
            serialized_mask[key] = [[value] for value in values]
        return serialized_mask


class BatchSampler(Sampler):
    """
        Object used to create batches of sequences of same length,
        highly inspired by: https://discuss.pytorch.org/t/tensorflow-esque-bucket-by-sequence-length/41284/9
    """

    def __init__(self, x, idx, batch_size):
        """
            Maps sequence indices to their length then generates the batches of sequences,
            each batch has only sequences of same length

            Args:
                x: (N, M) N sequences of different length having M features
                idx: indexes of the sequences in the dataset

        """
        super().__init__(None)
        self.batch_size = batch_size
        idx_seq_len = []
        shapes = []
        zero_x = []
        for i, p in enumerate(x):
            if p.shape[0] not in shapes:  # Add the sequence length to the list of sequences lengths
                shapes.append(p.shape[0])
            if p.shape[0] == 0:
                zero_x.append((idx[i], p.shape[0]))
            idx_seq_len.append((i, p.shape[0]))  # Map sequence indices to their length
        self.idx_seq_len = idx_seq_len
        self.batch_list = self._generate_batch_map()  # Generate
        self.num_batches = len(self.batch_list)

    def _generate_batch_map(self):
        """
            Generate the batches, each batch contains only sequences of same length
        """
        # shuffle all of the indices first so they are put into buckets differently
        shuffle(self.idx_seq_len)
        # Organize lengths, e.g., batch_map[10] = [30, 124, 203, ...] <= indices of sequences of length 10
        batch_map = OrderedDict()
        for idx, length in self.idx_seq_len:
            if length not in batch_map:
                batch_map[length] = [idx]
            else:
                batch_map[length].append(idx)
        # Use batch_map to split indices into batches of equal size
        # e.g., for batch_size=3, batch_list = [[23,45,47], [49,50,62], [63,65,66], ...]
        batch_list = []
        for length, indices in batch_map.items():
            for group in [indices[i:(i + min(self.batch_size, len(indices) - i))] for i in
                          range(0, len(indices), self.batch_size)]:
                batch_list.append(group)
        return batch_list

    def batch_count(self):
        return self.num_batches

    def __len__(self):
        return len(self.idx_seq_len)

    def __iter__(self):
        self.batch_list = self._generate_batch_map()
        # shuffle all the batches so they arent ordered by bucket size
        shuffle(self.batch_list)
        for i in self.batch_list:
            yield i


def sample_10folds_from_homr_experiment(test_obs: str,
                                        df_folds: pd.DataFrame) -> Dict:
    """
    Performs a 10 folds cross validation over the dataset to replicate the same folds used in the HOMR experiment

    Args:
        test_obs: Dataframe "int_test_observations.csv" with visits_ids in each test fold
        df_folds: Dataframe with visits between 01/07/2011 to 30/06/2016 with a column "fold"

    Returns: idx of train and test samples in each of the 10 folds
    """

    # Read dataframes
    test_obs = pd.read_csv(test_obs)
    masks = {}

    for i in range(10):
        # Get train samples indexes
        train_idx = df_folds.index[df_folds['fold'] != (i + 1)].tolist()
        # Get test visits
        test_visits = test_obs[test_obs['fold'] == (i + 1)]['visit_id']
        # Get test indexes
        test_idx = df_folds['visit_id'][df_folds['visit_id'].isin(test_visits)].index.tolist()

        if bool(set(train_idx) & set(test_idx)):
            raise ValueError(f'iteration {i}, intersection not null')

        masks[i] = {MaskType.TRAIN: train_idx, MaskType.VALID: None, MaskType.TEST: test_idx, MaskType.INNER: {}}

    return masks


def sample_unique_patients_for_train(map_idx: Dict,
                                     n: int,
                                     random_state: int,
                                     train_idx: List[int]
                                     ):
    """
    Samples a single visit (index) per patient in the train set we use to feed the RF_HOMR

    Args:
        map_idx: map of patient ids to all their indexes (visits)
        n: number of bootstraps
        random_state: to replicate the study
        train_idx: (M,) idx of the train set

    Returns: a list of size (N, M) where N is number of bootstraps
    """
    seed(random_state)
    bootstraps = [np.random.choice(map_idx[k], n)
                  for k in map_idx.keys()
                  # if one patient's visit is in the testing set then the others too since we split per patient
                  if map_idx[k][0] in train_idx]

    # Construct a list of size N x P where N is the number of bootstraps and P is the number of unique patients
    # The list contains the unique patient's indexes (unique patient's visit index) to consider in each bootstrap
    tuples = list(zip(*bootstraps))
    serialized_bootstraps = [list(t) for t in tuples]

    # Transform the list to a one hot encoded list of size (M,)
    one_hot_bootstraps = []
    for i in range(n):
        setx = set(serialized_bootstraps[i])
        one_hot_bootstraps.append([1 if x in setx else 0 for x in train_idx])

    return one_hot_bootstraps


def sample_multiple_visits_per_patient(df_folds: pd.DataFrame):
    """
    Performs a 10 folds cross validation over the dataset to replicate the same folds used in the HOMR experiment
    but all the visits of patients are included in the test set

    Args:
        df_folds: Dataframe with visits between 01/07/2011 to 30/06/2016 with a column "fold"

    Returns: idx of train and test samples in each of the 10 folds
    """
    masks = {}
    for i in range(10):
        # Get train samples indexes
        train_idx = df_folds.index[df_folds['fold'] != (i + 1)].tolist()
        test_idx = df_folds.index[df_folds['fold'] == (i + 1)].tolist()

        if bool(set(train_idx) & set(test_idx)):
            raise ValueError(f'iteration {i}, intersection not null')

        masks[i] = {MaskType.TRAIN: train_idx, MaskType.VALID: None, MaskType.TEST: test_idx, MaskType.INNER: {}}

    return masks


def unpack_temporal_visits_homr_experiment(masks: Dict,
                                           map_idx: Dict[int, Dict[int, int]]):
    """
    Unpack the test sets in the 10 folds cross validation of HOMR experiment so we can evaluate the
    model at different timestamps (1st visits, 2nd visits, .., last visits)

    Args:
        masks: 10 masks of training and testing as tested in the HOMR experiment
        map_idx: dictionary which maps each patient id to its indexes ordered in an increasing fashion
                example : {'patient_id': [1st_visit_index, 2nd_visit_index, ..., last_visit_index]}
    """

    for k in range(10):
        test_ids = []
        # Get patient's ids of the test set from the indexes
        for id_, indexes in map_idx.items():
            if set(indexes.values()).issubset(masks[k][MaskType.TEST]):
                test_ids.append(id_)

        for i in range(1, constants.MAX_VISIT + 1):
            # Get the testing set where each observation is at most the ith visit
            masks[k][MaskType.TEST + f"_{i}th_visit"] = [map_idx[_id][min(max(map_idx[_id].keys()), i)] for _id in
                                                         test_ids]

        # Get the testing set with all last visits
        masks[k][MaskType.TEST + '_last_visits'] = [map_idx[_id][max(map_idx[_id].keys())] for _id in test_ids]


def create_inner_splits_homr_experiment(masks: Dict,
                                        map_ids_idx: Dict,
                                        n_inner: int = 5,
                                        ):
    """
     Performs a K folds cross validation  to build testing, training and validation inner masks
     according to the outer masks

         Args:
             masks: masks containing outer splits
             map_ids_idx: dictionary mapping patient ids to their indexes
             n_inner: number of inner splits in each outer split

     Returns: a nested dictionary containing the train, test and validation masks in the inner and outer splits
    """

    for i, key in enumerate(masks.keys()):
        masks[i][MaskType.INNER] = {}
        # Perform a k fold cross validation on the training mask
        # Get train ids
        train_mask = masks[i][MaskType.TRAIN]
        train_ids = []
        for id_, indexes in map_ids_idx.items():
            if set(indexes).issubset(train_mask):
                train_ids.append(id_)

        inner_k_folds = KFold(n_splits=n_inner, random_state=101, shuffle=True)

        splits = inner_k_folds.split(train_ids)

        for j, (inner_remaining_idx, inner_test_masks) in enumerate(splits):
            # Get inner test ids
            inner_test_ids = np.array(train_ids)[inner_test_masks]
            inner_train_ids = np.array(train_ids)[inner_remaining_idx]

            # Get indexes:
            train_idx = []
            for _id in inner_train_ids:
                train_idx += map_ids_idx[_id]

            test_idx = []
            for _id in inner_test_ids:
                test_idx += map_ids_idx[_id]

            # Save the inner masks
            masks[i][MaskType.INNER][j] = {MaskType.TRAIN: train_idx,
                                           MaskType.VALID: None,
                                           MaskType.TEST: test_idx}
