## POYM : Prediction of One-Year Mortality risk 
This repository stores the code implemented to generate the results of the paper: *Leveraging patients’ longitudinal data to improve the Hospital One-year Mortality Risk* available as a preprint in medrXiv in: [https://doi.org/10.1101/2024.06.21.24309191](https://doi.org/10.1101/2024.06.21.24309191).
It utilizes data from patients admitted at The University Hospital of Sherbrooke, Canada, between July 1, 2011, and June 30, 2021, to predict their one-year mortality risk.

## 1. Data Availability
 **Please note that the hospitalization data used in this paper cannot be publicly shared due to regulations safeguarding patient privacy.** However, a synthetic dataset was generated using the [AVATAR method](https://doi.org/10.1038/s41746-023-00771-5) in partnership with [Octopize](https://www.octopize.io/), in which a synthetic patient is created for each patient from the original dataset. This synthetic dataset is publicly available on : [10.5281/zenodo.12954672](https://zenodo.org/doi/10.5281/zenodo.12954672). 
 
 To preserve patient's privacy, we did not save admission and discharge dates. Consequently, it is not possible to split the dataset temporally as done with the original dataset (**Section 3.3.2**) or to identify admissions with same-day discharge.

Below, we present the mean AUROC and the standard deviation of the ELSTM model from a 5-fold cross-validation over the entire dataset, comparing results obtained from the original data versus the synthetic dataset.

| Dataset Type | AdmDemo (Original) | AdmDemo (Synthetic) | AdmDemoDx (Original) | AdmDemoDx (Synthetic) |
|--------------|--------------------|---------------------|----------------------|-----------------------|
| Last visit  | 90.2 +- 0.3 | 90.8 +- 0.3 | 92.8 +- 0.2 | 92.9 +- 0.2 |
|Any visit     | 87.4 +- 0.2 | 88.0 +- 0.3 | 90.6 +- 0.1 | 90.7 +- 0.2 |


We encourage the use of the synthetic data to further research in this area. If you require access to the original dataset, please contact the corresponding author of the paper to initiate a Data Sharing Agreement.
## 2. How to use the package?
First, install the requirements under **Python 3.10.9** as following:
```
$ pip install -r requirements.txt
```
Download the synthetic dataset from [10.5281/zenodo.12954672](https://zenodo.org/doi/10.5281/zenodo.12954672) and move it to [csvs](csvs).

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
├── checkpoints                  <- Temporary state dictionaries save by the EarlyStopper module
├── csvs                         <- CSV files of the dataset used in the study
├── experiments                  <- Scripts to replicate our experiments
├── feature_importance           <- Directory to store the models performances for each group of patients after permutations
├── figures                      <- Figures generated in the visualization script
├── final_models                 <- Pretrained LSTMs with the entire learning set (see Section 4.2)
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
│   │   └── hyperparameters.py    <- Defines hyperparameters types
│   │   └── metric_scores.py      <- Custom metrics implementations and wrappers
│   │   └── utils.py              <- Functions associated to statistical tests and dataset anonymization
│   │   └── visualization.py      <- Functions to visualize different steps of the experiment
└── README.md
