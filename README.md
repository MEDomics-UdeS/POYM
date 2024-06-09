## POYM : Prediction of One-Year Mortality risk 
This repository stores the code implemented to generate the results of the paper: *Leveraging patients’ longitudinal data to improve the Hospital One-year Mortality Risk*.
It utilizes data from patients admitted at The University Hospital of Sherbrooke, Canada, between July 1, 2011, and June 30, 2021, to predict their one-year mortality risk. **Please note that the hospitalization data used in this paper cannot be publicly shared due to regulations safeguarding patient privacy.**

## 1. Data Availability
To test the code implemented in this work, we have created two alternative datasets:

1. **Random Dataset**: Each patient in the dataset is matched to a randomly generated patient using a categorical distribution on each predictor. This random dataset is publicly available [here](https://drive.google.com/file/d/1VjzDgbkeob50ZV1RSuzrmWG63VI4yWjM/view?usp=sharing).
2. **Synthetic Dataset**: Each patient in the dataset is matched to a synthetic patient generated using the [AVATAR method](https://doi.org/10.1038/s41746-023-00771-5). We are currently working on obtaining ethical approval to share this synthetic dataset publicly.

For both datasets, we did not save admission and discharge dates to preserve patient's privacy. Consequently, it is not possible to split the dataset temporally as done with the original dataset (**Section 3.3.2**) or to identify admissions with same-day discharge.

Below, we present the mean AUROC and the standard deviation of the ELSTM model from a 5-fold cross-validation over the entire dataset, comparing results obtained from the original data versus the synthetic dataset.

| Dataset Type | AdmDemo (Original) | AdmDemo (Synthetic) | AdmDemoDx (Original) | AdmDemoDx (Synthetic) |
|--------------|--------------------|---------------------|----------------------|-----------------------|
| Last visit  | 90.2 +- 0.3 | 90.8 +- 0.3 | 92.8 +- 0.2 | 92.9 +- 0.2 |
|Any visit     | 87.4 +- 0.2 | 88.0 +- 0.3 | 90.6 +- 0.1 | 90.7 +- 0.2 |


We encourage the use of the synthetic data to further research in this area. If you require access to the original dataset, please contact [Martin Vallières](martin.vallieres@usherbrooke.ca) to initiate a Data Sharing Agreement.
## 2. How to use the package?
First, install the requirements under **Python 3.10.9** as following:
```
$ pip install -r requirements.txt
```
Move the dowloaded dataset to [csvs](csvs).
To perform model comparaisons experiments, run:
```
$ python experiments/model_selection.py
```
To replicate the final results on eligible end-of-life care patients, run:
```
$ python experiments/final_validation.py
```
To compute feature importance on the final ELSTM model using AdmDemoDx predictors, run:
```
$ python experiments/feature_importance.py
```
## Project Tree
```
├── csvs                         <- CSV files of the dataset used in the study
├── hps                          <- Python file to store the range of hyperparameters values
├── settings                     <- Files for project setup
├── src                          <- All project modules
│   ├── data
|   │   ├── processing
|   │   │   ├── constants.py          <- Constants related to the study
|   │   │   ├── datasets.py           <- Custom dataset implementation for the study
|   │   │   └── preparing.py          <- Cleaning and data preparation
|   │   │   └── sampling.py           <- Sampling of the dataset
|   │   │   └── transforms.py         <- Data preprocessing
│   ├── evaluation
│   │   ├── early_stopping.py     <- Module for early stopping on validation data
│   │   └── evaluating.py         <- Skeleton of each experiment process
│   │   └── tuning.py             <- Hyper-parameters optimizations 
│   └── utils                     
│   │   └── delong.py             <- Fast implementation of DeLong test
│   │   └── hyperparameters.py    <- Defines hyperparameters types
│   │   └── metric_scores.py      <- Custom metrics implementations and wrappers
│   │   └── visualization.py      <- Functions to visualize different steps of the experiment
├── model_selection.py            <- Main script to perform model comparaison
├── final_validation.py           <- Main script to perform the final validation of LSTM model
├── feature_importance.py         <- Main script to measure feature importance on the final LSTM model
└── README.md
