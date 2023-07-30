"""
Filename: transforms.py

Author: Nicolas Raymond

Description: Defines all transformations related to preprocessing treatment

Date of last modification : 2021/11/01
"""

from typing import Optional, Tuple, List

import pandas as pd
from torch import from_numpy, tensor


class ContinuousTransform:
    """
    Class of transformations that can be applied to continuous data
    """

    @staticmethod
    def normalize(df: pd.DataFrame,
                  mean: Optional[pd.Series] = None,
                  std: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Applies normalization to columns of a pandas dataframe
        """
        if mean is not None and std is not None:
            return (df-mean)/std
        else:
            return (df-df.mean())/df.std()

    @staticmethod
    def standardize(df: pd.DataFrame,
                    interval: List[int],
                    minima: Optional[int] = None,
                    maxima: Optional[int] = None
                    ) -> pd.DataFrame:
        """
        Applies MaxScaling standardization to columns of a pandas dataframe, values are in [0, interval]
        """
        interval = 4
        if minima is not None and maxima is not None:
            return (df-minima)/(maxima-minima)*interval
        else:
            return (df-df.min())/(df.max()-df.min())*interval

    @staticmethod
    def to_tensor(df: pd.DataFrame) -> tensor:
         return from_numpy(df.to_numpy(dtype=float)).float()


class CategoricalTransform:
    """
    Class of transformation that can be applied to categorical data
    """

    @staticmethod
    def one_hot_encode(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        One hot encodes all columns of the dataframe

        Returns : the dataframe encoded and its new columns
        """
        df = pd.get_dummies(df)
        return df, list(df.columns)

    @staticmethod
    def ordinal_encode(df: pd.DataFrame,
                       encodings: Optional[dict] = None) -> Tuple[pd.DataFrame, dict]:
        """
        Applies ordinal encoding to all columns of the dataframe
        """
        if encodings is None:
            encodings = {}
            for c in df.columns:
                encodings[c] = {v: k for k, v in enumerate(df[c].cat.categories)}
                df[c] = df[c].cat.codes

        else:
            for c in df.columns:
                column_encoding = encodings[c]
                df[c] = df[c].apply(lambda x: column_encoding[x])

        return df, encodings

    @staticmethod
    def to_tensor(df: pd.DataFrame) -> tensor:
        """
        Takes a dataframe with categorical columns and returns a tensor with "longs"

        Args:
            df: dataframe with categorical columns only

        Returns: tensor
        """
        return from_numpy(df.to_numpy(dtype=float)).long()

