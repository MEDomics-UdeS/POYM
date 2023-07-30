# Pre-release for T-CAIREM AI in Medicine Conference
## POYM : Patient One-Year Mortality risk prediction
This package stores the code used to generate the results of our experiments. It utilizes data from patients admitted at The University Hospital of Sherbrooke, Canada, between July 1, 
2011, and June 30, 2021, to predict one-year mortality risk. Please note that the hospitalization data used in this article cannot be publicly shared due to regulations safeguarding patient privacy.

## 1. How to use the package?
First, install the requirements under **Python 3.10.9** as following:
```
$ pip install -r settings/requirements.txt
```
To perform model comparaisons experiments, run 
```
$ python model_selection.py
```
To replicate the final results on the eligible end-of-life care patients, run 
```
$ python final_validation.py
```
To compute feature importance on the final LSTM model, run:
```
$ python feature_importance.py
```
## Project Tree
```
├── csvs                         <- CSV files of the dataset used in the study
├── hps                          <- Python file to store range of hyperparameters values
├── settings                     <- Files to set the project setup
├── src                          <- All project modules
│   ├── data
|   │   ├── processing
|   │   │   ├── constants.py          <- Constants related to the study
|   │   │   ├── datasets.py           <- Custom dataset implementation for the study
|   │   │   └── preparing.py          <- Cleaning and data preparation
|   │   │   └── sampling.py           <- Sampling of the dataset
|   │   │   └── transforms.py         <- Data preprocessing
│   ├── evaluation
│   │   ├── early_stopping.py     <- Module to perform early stopping on validation data
│   │   └── evaluating.py         <- Skeleton of each experiment process
│   │   └── tuning.py             <- Hyper-parameters optimizations using different optimizers
│   └── utils                     
│   │   └── delong.py             <- Fast implementation of DeLong test
│   │   └── hyperparameters.py    <- Defines hyperparameters types
│   │   └── metric_scores.py      <- Custom metrics implementations and wrappers
│   │   └── visualization.py      <- Functions to visualize different steps of the experiment
├── model_selection.py            <- Main script to perform model comparaison
├── final_validation.py           <- Main script to perform the final validation of LSTM model
├── feature_importance.py         <- Main script to measure feature importance on the final LSTM model
└── README.md
