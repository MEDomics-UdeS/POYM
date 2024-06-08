"""
Filename: preparing.py

Authors: Hakima Laribi

Description: Defines the DataPreparer used to prepare the different cohorts of the project

"""
import numpy as np
import pandas as pd
import json

from sklearn.model_selection import train_test_split

from src.data.processing import constants


class DataPreparer:
    """
    Object used to prepare dataframes used for our learning task.

    It selects the predictors used for our learning task and renames values of some variables
    """

    def __init__(self,
                 train_file: str = "csvs/df_train.csv",
                 test_file: str = None,
                 task: str = constants.OYM,
                 split_train_test: int = None
                 ):
        """
        Saves private attributes

        Args:
            train_file : path to the csv of the training cohort with visits between 01-07-2011 and 30-06-2017
            test_file : path to the csv of the testing cohort with visits between 01-07-2017 and 30-06-2021
            task : name of the outcome column
            split_train_test: if not None, split the train_file to train and test sets with split_train_test elements
            in the training set
        """

        # Internal private fixed attributes
        self.task = task

        if split_train_test is not None and test_file == None:
            df = pd.read_csv(train_file)
            # Divide dataset to learning set and holdout set according to admission dates
            if "admission_date" in df.columns:
                df['admission_date'] = pd.to_datetime(df['admission_date'])
                df.sort_values(by='admission_date', inplace=True)
                self.__training_cohort = df[df['admission_date'] < '2017-07-01']
                holdout = df[df['admission_date'] >= '2017-07-01']
                # Remove patients admitted before 2017-07-01 to avoid data leakage between both sets
                self.__testing_cohort = holdout[~holdout['patient_id'].isin(self.__training_cohort['patient_id'])]

            # Divide dataset to learning set and holdout set randomly
            else:
                train_patients, test_patients = train_test_split(df['patient_id'].unique(),
                                                                 train_size=split_train_test,
                                                                 random_state=101)
                self.__training_cohort = df[df['patient_id'].isin(train_patients)]
                self.__testing_cohort = df[df['patient_id'].isin(test_patients)]

        else:
            self.__training_cohort = pd.read_csv(train_file)
            self.__testing_cohort = pd.read_csv(test_file) if test_file is not None else None

        # select the predictors, the visit_id and the patient_id for each cohort and the target column
        self.__training_cohort = self.select_variables(self.__training_cohort)

        # add ID column to the training set
        self.__training_cohort = self.add_ids(self.__training_cohort)

        if self.__testing_cohort is not None:
            self.__testing_cohort = self.select_variables(self.__testing_cohort)

            # add ID column to the training set
            self.__testing_cohort = self.add_ids(self.__testing_cohort, self.__training_cohort['ids'].max() + 1)

        # Digitize target variable and make sure the categorical columns_to_anonymize are in the right format
        for df in [self.__training_cohort, self.__testing_cohort]:
            if df is not None:
                df[self.task] = np.where(df[self.task], 1, 0)
                df[constants.CAT_COLS] = df[constants.CAT_COLS].astype('category')
                # change values of the columns_to_anonymize : ( gender, living_status, admission_group, service_group)
                for column in constants.COL_VALUES_RENAMED:
                    self.rename_column_values(df, column)

                # Add the rank of each visit, visits are ordered in the dataset according to their occurrences
                if "nb_visits" not in df.columns:
                    df.sort_values(by='patient_id', inplace=True)
                    pt_groups = df.groupby('patient_id')

                    # Process visits and concatenate DataFrames
                    rank_visits = []
                    for _, visits in pt_groups:
                        rank_visits += list(range(1, len(visits) + 1))

                    df["nb_visits"] = rank_visits

    @property
    def get__training_cohort(self):
        df = self.__training_cohort.reset_index(drop=True)
        return df

    @property
    def get__testing_cohort(self):
        return self.__testing_cohort.reset_index(drop=True)

    def create_CDSS_eligble_cohort(self,
                                   df: pd.DataFrame = None) -> pd.DataFrame:
        """
            Selects eligible hospitalizations for CDSS (Clinical Decision Support System)
            as done in : 10.1093/jamia/ocab140
        """
        df = self.__testing_cohort if df is None else df
        if (
                all(item in df.columns for item in
                    ["discharge_date", "admission_date"])):

            return df[(df['service_group'] != "Palliative care") &
                      (df['service_group'] != "Obstetrics") &
                      (df['admission_group'] != "Obstetrics") &
                      (((pd.to_datetime(df['discharge_date'], format='%Y-%m-%d')
                         -
                         pd.to_datetime(df['admission_date'], format='%Y-%m-%d')).dt.days)
                       > 0)
                      ]
        else:
            return df[(df['service_group'] != "Palliative care") &
                      (df['service_group'] != "Obstetrics") &
                      (df['admission_group'] != "Obstetrics")
                      ]

    def get_cdss_eligible_idx(self, df: pd.DataFrame) -> np.array:
        """
            Get ids of patients eligible to end-of-life care
        """
        f = self.create_CDSS_eligble_cohort(df)
        return np.array(f['ids'])

    def get_cdss_eligible_indexes(self, df: pd.DataFrame):
        """
            Get indexes of visits eligible to end-of-life care
        """
        f = self.create_CDSS_eligble_cohort(df)
        return f.index.to_list()

    def select_variables(self,
                         df: pd.DataFrame,
                         fold: bool = False) -> pd.DataFrame:
        """
        Select the patient_id, visit_id, 244 predictors used in HOMR model, the target variable (One-year Mortality)
        and the fold where each visit is affected if fold is set to true

            Args:
                df : pandas dataframe
                fold : boolean to specify if we want to select variables of the training set or the testing set

        """
        for item in [constants.PATIENT] + [constants.VISIT] + constants.PREDICTORS + [self.task]:
            if item not in df.columns:
                raise Exception(f"Dataframe has missing necessary columns: {item}")

        if fold:
            if constants.FOLD not in df.columns:
                raise Exception("Training set must have column FOLD")
            return df

        else:
            return df

    @staticmethod
    def add_ids(df: pd.DataFrame,
                shift: int = 0):
        """
        Add a column IDS to the dataframe

        Args:
            df : pandas dataframe
            shift: start number from which identifiers are generated
        """
        patients = df[constants.PATIENT].unique()
        k = pd.DataFrame({constants.IDS: list(range(shift, len(patients) + shift)),
                          constants.PATIENT: patients})
        df = pd.merge(df, k, on=constants.PATIENT)
        return df.reset_index(drop=True)

    @staticmethod
    def rename_column_values(
            df: pd.DataFrame,
            column: str) -> pd.DataFrame:
        """
        Renames values of a specific column

        Args:
            df : pandas dataframe
            column : name of the column with values to rename
        """
        if column not in constants.COL_VALUES_RENAMED:
            raise ValueError(
                f"Column with values to rename must be in : {constants.COL_VALUES_RENAMED}")

        else:
            if column == "gender":
                df["gender"] = np.where(df['gender'] == "M", 'Male', 'Female')

            elif column == "living_status":
                case_when = [
                    df["living_status"] == "chronic_care",
                    df["living_status"] == "home",
                    df["living_status"] == "nursing_home",
                    df["living_status"] == "unknown"
                ]
                do = [
                    "Chronic care hospital",
                    "Home",
                    "Nursing home",
                    "Unknown (prospectively inaccessible)"
                ]
                df["living_status"] = np.select(case_when, do)

            elif column == "admission_group":
                case_when = [
                    df["admission_group"] == "elective",
                    df["admission_group"] == "obstetrics",
                    df["admission_group"] == "semiurgent",
                    df["admission_group"] == "urgent"
                ]
                do = [
                    "Elective",
                    "Obstetrics",
                    "Semi-urgent",
                    "Urgent"
                ]
                df["admission_group"] = np.select(case_when, do)
            else:
                case_when = [
                    df["service_group"] == "cardiology",
                    df["service_group"] == "cardiovascular_surgery",
                    df["service_group"] == "endocrinology",
                    df["service_group"] == "family_medicine",
                    df["service_group"] == "gastroenterology",
                    df["service_group"] == "general_surgery",
                    df["service_group"] == "gynecology",
                    df["service_group"] == "hematology_oncology",
                    df["service_group"] == "icu",
                    df["service_group"] == "internal_medicine",
                    df["service_group"] == "maxillo_surgery",
                    df["service_group"] == "nephrology",
                    df["service_group"] == "neurology",
                    df["service_group"] == "neurosurgery",
                    df["service_group"] == "obstetrics",
                    df["service_group"] == "opthalmology",
                    df["service_group"] == "ORL",
                    df["service_group"] == "orthopedic_surgery",
                    df["service_group"] == "palliative_care",
                    df["service_group"] == "plastic_surgery",
                    df["service_group"] == "respirology",
                    df["service_group"] == "rhumatology",
                    df["service_group"] == "thoracic_surgery",
                    df["service_group"] == "trauma",
                    df["service_group"] == "urology",
                    df["service_group"] == "vascular_surgery",
                    df["service_group"] == "interventional_radiology",
                ]
                do = [
                    "Cardiology",
                    "Cardiac surgery",
                    "Endocrinology",
                    "Family medicine",
                    "Gastro-enterology",
                    "General surgery",
                    "Gynecology",
                    "Hematology / Oncology",
                    "Critical care",
                    "Internal medicine",
                    "Maxillo-facial surgery",
                    "Nephrology",
                    "Neurology",
                    "Neurosurgery",
                    "Obstetrics",
                    "Opthalmology",
                    "Otorhinolaryngology",
                    "Orthopedic surgery",
                    "Palliative care",
                    "Plastic surgery",
                    "Respirology",
                    "Rhumatology",
                    "Thoracic surgery",
                    "Trauma",
                    "Urology",
                    "Vascular surgery",
                    "Interventional radiology"
                ]
                df["service_group"] = np.select(case_when, do)
        return df
