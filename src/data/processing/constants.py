"""
Filename: constants.py

Authors: Hakima Laribi

Description: Defines constants related to the dataset
"""

import re

import pandas as pd

# Read full dataset
df = pd.read_csv("csvs/df_train.csv")

# Comorbidities diagnostic variables
DX_COLS = [dx for dx in list(df.columns) if re.compile("^(dx_).+").search(dx)]

# Admission diagnosis variables
ADM_COLS = [adm for adm in list(df.columns) if re.compile("^(adm_).+").search(adm)]

# Demographic, previous care utilization and characteristics of the current admission variables
OTHER_COLS = [
    "age_original",
    "gender",
    "ed_visit_count",
    "ho_ambulance_count",
    "flu_season",
    "living_status",
    "total_duration",
    "admission_group",
    "is_ambulance",
    "is_icu_start_ho",
    "is_urg_readm",
    "service_group",
    "has_dx"
]
# Target variable
OYM = "oym"

# Fold variable
FOLD = "fold"

# Patient id
PATIENT = "patient_id"

# Visit id
VISIT = "visit_id"

# ID of the observation

IDS = "ids"
# Continious variables
CONT_COLS = [
    "age_original",
    "ed_visit_count",
    "ho_ambulance_count",
    "total_duration"
]

CONT_COLS_ADM = [
    "age_original",
]

# Categorical variables
CAT_COLS = [cat for cat in DX_COLS + ADM_COLS + OTHER_COLS if cat not in CONT_COLS]

OTHER_CAT_COL = [cat for cat in CAT_COLS if cat not in DX_COLS + ADM_COLS + ["has_dx"]]

# 244 PREDICTORS of HOMR
PREDICTORS = OTHER_COLS + DX_COLS + ADM_COLS

# Columns with values to rename
COL_VALUES_RENAMED = [
    "gender",
    "living_status",
    "admission_group",
    "service_group",
]

# Maximum number of visits to consider
MAX_VISIT = 5
