"""
Filename: datasets.py

Description: Defines the classes related to datasets
"""

from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import torch
from numpy import array, concatenate
from pandas import DataFrame, Series
from torch import cat, from_numpy, tensor
from torch.utils.data import Dataset

from src.data.processing import constants
from src.data.processing.transforms import ContinuousTransform, CategoricalTransform


class Encodings:
    """
    Stores the constant related to encoding type
    """
    ONE_HOT: str = "one hot"
    ORDINAL: str = "ordinal"

    def __iter__(self):
        return iter([self.ONE_HOT, self.ORDINAL])


class HOMRDataset(Dataset):
    """
    Custom dataset class for HOMR experiments
    """

    def __init__(self,
                 dataset: pd.DataFrame,
                 target: str,
                 ids: str,
                 cont_cols: Optional[List[str]] = None,
                 cat_cols: Optional[List[str]] = None,
                 encoding: Optional[str] = None,
                 to_tensor: bool = False,
                 norm_col: Optional[List[str]] = None,
                 temporal: bool = False,
                 super_learning: bool = False
                 ):
        """
        Sets protected and public attributes of our custom dataset class

        Args:
            dataset: dataframe with the original data
            target: name of the column with the targets
            ids : name of the column with participant ids
            cont_cols: list of column names associated with continuous data
            cat_cols: list of column names associated with categorical data
            to_tensor: true if we want the features and targets in tensors, false for numpy arrays
            norm_col: columns to normalize, if none, normalize all continuous columns
            temporal: specifies if the dataset will be used for a temporal analysis
            super_learning: specifies if the dataset will be used for super learning purposes

        """
        # Validations of inputs
        if constants.IDS not in dataset.columns:
            raise ValueError("Hospitalisations' ids missing from the dataframe")

        if cont_cols is None and cat_cols is None:
            raise ValueError("At least a list of continuous columns or a list of categorical columns must be provided.")

        if encoding is not None and encoding not in Encodings():
            raise ValueError(f"{encoding} can't be resolved to an encoding type")

        # Set default protected attributes
        self._cat_cols, self._cat_idx = cat_cols, []
        if norm_col is not None:
            self._non_norm_cols = [col for col in cont_cols if col not in norm_col]
            self._cont_cols = norm_col + self._non_norm_cols
        else:
            self._cont_cols = cont_cols
        self._cont_idx = []
        self._binary_col = [col for col in cat_cols if (len(np.unique(dataset[col])) == 2) &
                            (0 in np.unique(dataset[col]))
                            & (1 in np.unique(dataset[col]))] if cat_cols is not None else None
        self._norm_col = norm_col
        self.__id_column = ids
        self._ids = list(dataset[ids].values)
        self._n = dataset.shape[0]
        self._original_data = dataset
        self._encoded_data = dataset
        self._target = target
        self._to_tensor = to_tensor
        self._encoding, self._encodings = encoding, {}
        self._temporal_analysis = temporal
        self._super_learner = super_learning

        # Initialize train, valid and test masks
        self._train_mask, self._valid_mask, self._test_mask = [], None, []

        # Get categorical and continuous datasets in the appropriate container and update categorical and continuous idx
        self._x_cat, self._x_cont = self._get_categorical_set(), self._get_continuous_set()

        # Define protected attribute for targets
        self._y = self._initialize_targets(dataset[target], to_tensor)

        # Define protected attribute getter for the whole dataset
        self._x = self._fill_dataset

        # Initialize statistics related to the dataset
        self._modes, self._mu, self._std = None, None, None

        # We update current training mask with all the data
        self.update_masks(list(range(self._n)), [], [])

    def __len__(self):
        return self._n

    def __getitem__(self,
                    idx: Union[int, List[int], List[List[int]]]
                    ) -> Union[Tuple[array, array, array], Tuple[tensor, tensor, tensor]]:
        if self._temporal_analysis & (isinstance(idx, list)) & (isinstance(idx[0], list)):
            if isinstance(idx, list):
                # Squeeze the idx
                squeezed_idx = []
                for indexes in idx:
                    squeezed_idx += indexes
                x, y = self.x[squeezed_idx], self.y[squeezed_idx]

                initial_id = 0
                reshaped_x = []
                reshaped_y = []
                # Put x and y in the same shapes as idx
                for i, indexes in enumerate(idx):
                    last_id = len(indexes) if isinstance(indexes, list) else 1
                    reshaped_x.append(x[initial_id:initial_id + last_id])
                    reshaped_y.append(y[initial_id:initial_id + last_id])
                    initial_id += last_id
                return reshaped_x, reshaped_y, idx
            else:
                return [self.x[idx]], [self.y[idx]], idx
        else:
            return self.x[idx], self.y[idx], idx

    @property
    def cat_cols(self) -> List[str]:
        return self._cat_cols

    @property
    def cat_idx(self) -> List[int]:
        return self._cat_idx

    @property
    def cat_sizes(self) -> Union[List[int], None]:
        if self._encoding is Encodings.ORDINAL:
            return [len(np.unique(self._original_data[c])) for c in self._cat_cols]
        else:
            # one hot encoding
            return [2] * len(self.cat_cols)

    @property
    def cont_cols(self) -> List[str]:
        return self._cont_cols

    @property
    def cont_idx(self) -> List[int]:
        return self._cont_idx

    @property
    def encodings(self) -> Optional[Dict[str, Dict[str, int]]]:
        if self._encoding is not Encodings.ORDINAL:
            return None
        return self._encodings

    @property
    def ids(self) -> List[str]:
        return self._ids

    @property
    def original_data(self) -> DataFrame:
        return self._original_data

    @property
    def target(self) -> str:
        return self._target

    @property
    def test_mask(self) -> List[int]:
        return self._test_mask

    @property
    def train_mask(self) -> List[int]:
        return self._train_mask

    @property
    def valid_mask(self) -> Optional[List[int]]:
        return self._valid_mask

    @property
    def x(self):
        if torch.is_tensor(self._x):
            return self._x
        return self._x()

    @property
    def x_cat(self) -> Optional[Union[array, tensor]]:
        return self._x_cat

    @property
    def x_cont(self) -> Optional[Union[array, tensor]]:
        return self._x_cont

    @property
    def y(self) -> array:
        return self._y

    @property
    def encoded_data(self) -> pd.DataFrame:
        return self._encoded_data

    @property
    def mu(self) -> Optional[Series]:
        return self._mu

    @property
    def std(self) -> Optional[Series]:
        return self._std

    @property
    def modes(self) -> Optional[Series]:
        return self._modes

    @property
    def temporal_analysis(self) -> bool:
        return self._temporal_analysis

    def _get_categorical_set(self) -> Optional[Union[array, tensor]]:
        """
        Encode the dataset if specified then gets the categorical data of all observations in the original dataset
        and sets categorical columns idx

        Returns: array or tensor
                """
        if self._cat_cols is None:
            return []

        # Make sure that categorical data in the original dataframe is in the correct format
        self._original_data[self._cat_cols] = self._original_data[self._cat_cols].astype('category')

        # Get the original categorical non binary data
        col_to_encode = [col for col in self._cat_cols if col not in self._binary_col]
        dataset = self._original_data[col_to_encode]

        # One hot encode all the dataframe and get the new binary columns
        if self._encoding == Encodings.ONE_HOT:
            if len(self._binary_col) < len(self._cat_cols):
                dataset, cols_encoded = CategoricalTransform.one_hot_encode(dataset)
                self._cat_cols = self._binary_col + cols_encoded

        # Apply ordinal encoding to the whole dataset
        elif self._encoding == Encodings.ORDINAL:
            dataset, self._encodings = CategoricalTransform.ordinal_encode(dataset)
            for col in self._binary_col:
                self._encodings[col] = {0: 0, 1: 1}
            # Reorder the categorical columns
            self._cat_cols = self._binary_col + col_to_encode

        # Get the categorical dataset with binary and non-binary columns
        dataset = self._original_data[self._binary_col].join(dataset)

        # Join the encoded categorical set with the continuous set and save it
        if self.cont_cols is not None:
            self._encoded_data = self._original_data[self._cont_cols].join(dataset)
        else:
            self._encoded_data = dataset

        if self._cont_cols is None:
            # Only categorical column idx
            self._cat_idx = list(range(len(self._cat_cols)))

        else:
            nb_cont_cols = len(self._cont_cols)
            self._cat_idx = list(range(nb_cont_cols, nb_cont_cols + len(self._cat_cols)))

        # Return the data in the appropriate container
        x = np.array(dataset, dtype=np.float16)
        if self._to_tensor:
            return tensor(x, dtype=torch.float16)
        else:
            return x

    def _get_continuous_set(self) -> Optional[Union[array, tensor]]:
        """
        Gets the continuous data of all observations in the original dataset and sets continuous columns idx

        Returns: array or tensor

        """
        if self._cont_cols is None:
            return None

        self._cont_idx = list(range(len(self._cont_cols)))

        # get the continuous data
        dataset = self._original_data[self._cont_cols]

        x = np.array(dataset, dtype=np.float16)
        if self._to_tensor:
            return tensor(x, dtype=torch.float16)
        else:
            return x

    def _fill_dataset(self) -> Union[tensor, array]:
        """
        Fill the dataset with the categorical and continuous data of all observations

        Returns: array or tensor

        """
        if self._cat_cols is not None and self._cont_cols is not None:
            if self._to_tensor:
                return cat((self.x_cont, self.x_cat), dim=1)
            else:
                return concatenate((self.x_cont, self.x_cat), axis=1)
        elif self._cat_cols is not None:
            return self._x_cat
        else:
            return self._x_cont

    def update_masks(self,
                     train_mask: Union[List[int], List[List[int]], Dict[int, List[List[int]]]],
                     test_mask: Union[List[int], List[List[int]], Dict[int, List[List[int]]]],
                     valid_mask: Optional[Union[List[int], List[List[int]]], Dict[int, List[List[int]]]] = None,
                     graph_construction: bool = True) -> None:
        """
        Updates the train, valid and test masks

        Args:
            train_mask: list of idx in the training set
            test_mask: list of idx in the test set
            valid_mask: list of idx in the valid set
            graph_construction: boolean to specify whether we will construct the testing, training, validation graphs
            or not.

        Returns: None
        """

        # We set the new masks values
        self._train_mask, self._test_mask = train_mask, test_mask
        self._valid_mask = valid_mask if valid_mask is not None else []

        if not self._super_learner:
            # Compute the current values of modes, mu, std
            if self._norm_col is not None:
                self._modes, self._mu, self._std = self._current_train_stats(cont_cols=self._norm_col)
            else:
                self._modes, self._mu, self._std = self._current_train_stats()

            # Normalize continuous data according to the current training set statistics
            self._x_cont = self.numerical_setter(self._mu, self._std)

    def numerical_setter(self,
                         mu: float,
                         std: float) -> Union[tensor, array]:
        """
        Transform continuous columns with normalization

        Args:
            mu: mean
            std: standard variation

        Returns: tensor or array
        """
        if self._cont_cols is None:
            return None

        # Normalize the original data
        if self._norm_col is not None:
            x_norm = ContinuousTransform.normalize(self.original_data[self._norm_col], mu, std)
            x_non_norm = self.original_data[self._non_norm_cols]
            x_cont = x_norm.join(x_non_norm)

        else:
            x_cont = ContinuousTransform.normalize(self.original_data[self.cont_cols], mu, std)

        # Return the data in the appropriate container
        x = np.array(x_cont, dtype=np.float16)
        if self._to_tensor:
            return tensor(x, dtype=torch.float16)
        else:
            return x

    def _current_train_stats(self,
                             cat_cols: Optional[List[str]] = None,
                             cont_cols: Optional[List[str]] = None
                             ) -> Tuple[Optional[Series], Optional[Series], Optional[Series]]:
        """
        Compute statistics related to the current training set

        Args:
            cat_cols : categorical columns names
            cont_cols : continuous columns names

        Returns: modes, means and the standard deviation of each categorical and continuous column
        """

        # Input validation
        if cat_cols is not None and not (all(item in self._cat_cols for item in cat_cols)):
            raise ValueError("Selected categorical columns must exit in the original dataframe")

        if cont_cols is not None and not (all(item in self._cont_cols for item in cont_cols)):
            raise ValueError("Selected continuous columns must exit in the original dataframe")

        if self._temporal_analysis & (isinstance(self._train_mask[0], list)):
            train_mask = []
            for indexes in self._train_mask:
                train_mask += indexes
        else:
            train_mask = self._train_mask

        # Get the current training set from the encoded set
        train_data = self._encoded_data.iloc[train_mask]

        # Get the modes of each categorical column
        if cat_cols is None and self._cat_cols is None:
            modes = None
        elif cat_cols is not None:
            modes = train_data[cat_cols].mode().iloc[0]
        else:
            modes = train_data[self.cat_cols].mode().iloc[0]

        # Get the current training set from the original set
        train_data = self._original_data.iloc[train_mask]

        # Get the mean and standard deviation of each continuous column
        if cont_cols is None and self._cont_cols is None:
            mu, std = None, None
        elif cont_cols is not None:
            mu, std = train_data[cont_cols].mean(), train_data[cont_cols].std()
        else:
            mu, std = train_data[self.cont_cols].mean(), train_data[self.cont_cols].std()

        return modes, mu, std

    def _retrieve_subset_from_original(self,
                                       cont_cols: Optional[List[str]] = None,
                                       cat_cols: List[str] = None) -> DataFrame:
        """
        Returns a copy of a subset of the original dataframe

        Args:
            cont_cols: list of continuous columns
            cat_cols: list of categorical columns

        Returns: dataframe
        """
        selected_cols = []
        if cont_cols is not None:
            selected_cols += cont_cols
        if cat_cols is not None:
            selected_cols += cat_cols

        return self.original_data[[self.__id_column, self._target] + selected_cols].copy()

    def create_subset(self,
                      cont_cols: Optional[List[str]] = None,
                      cat_cols: List[str] = None) -> Any:
        """
        Returns a subset of the current dataset using the given cont_cols and cat_cols

        Args:
            cont_cols: list of continuous columns
            cat_cols: list of categorical columns

        Returns: instance of the PetaleDataset class
        """
        subset = self._retrieve_subset_from_original(cont_cols, cat_cols)
        return HOMRDataset(dataset=subset,
                           target=self.target,
                           ids=self.__id_column,
                           cont_cols=cont_cols,
                           cat_cols=cat_cols,
                           encoding=self._encoding,
                           to_tensor=self._to_tensor)

    @staticmethod
    def _initialize_targets(targets_column: Series,
                            target_to_tensor: bool
                            ) -> Union[array, tensor]:
        """
        Sets the targets according to the choice of container

        Args:
            targets_column: column of the dataframe with the targets
            target_to_tensor: true if we want the targets to be in a tensor, false for numpy array

        Returns: targets
        """
        # Set targets protected attribute according to task
        t = targets_column.to_numpy(dtype=float)
        if target_to_tensor:
            t = from_numpy(t).long()
        else:
            t = t.astype(int)

        return t.squeeze()

    def map_ids_to_indexes(self) -> Dict[int, Dict[int, int]]:
        """
            Maps each patient id to a map of (number of the visit, index)
        """
        map_ids = {}
        ids = self._original_data[self.__id_column].tolist()

        for id_ in ids:
            indexes = self.original_data.index[self.original_data[self.__id_column] == id_].tolist()
            id_obs = self.original_data.nb_visits[self.original_data[self.__id_column] == id_].tolist()
            map_ids[id_] = dict(zip(id_obs, indexes))

        return map_ids

    def map_ids_to_indexes_homr(self) -> Dict[int, int]:
        """
            Maps each patient id to the indexes of each of its visits
        """
        map_ids = {}
        ids = self._original_data[self.__id_column].tolist()

        for id_ in ids:
            indexes = self.original_data.index[self.original_data[self.__id_column] == id_].tolist()
            map_ids[id_] = indexes

        return map_ids

    def map_indexes_to_ids(self):
        """
            Maps each index to patient id
        """
        index_to_id = {}
        for idx, row in self._original_data.iterrows():
            index_to_id[idx] = row[self.__id_column]

        return index_to_id


class LightHOMRDataset(Dataset):
    """
        Light dataset class for HOMR experiments containing only data samples, classification labels
        and samples idx
    """

    def __init__(self,
                 x: Union[tensor, array],
                 y: Union[tensor, array],
                 idx: Union[tensor, array],
                 temporal_analysis: bool):
        """
        Sets protected and public attributes of our custom dataset class

        Args:
            x : (N,D) tensor or array with D-dimensional samples
            y : (N,) tensor or array with classification labels
            idx : (N,) tensor or array with idx of samples according to the whole dataset

        """
        # Sets the public attributes
        self.x = x
        self.y = y
        self.ids = idx
        self.temporal_analysis = temporal_analysis

    def __len__(self):
        return len(self.y)

    def __getitem__(self, indices: Union[int, List[int], List[List[int]]]
                    ) -> Union[Tuple[array, array, List], Tuple[tensor, tensor, List]]:
        return self.x[indices], self.y[indices], self.ids[indices]
