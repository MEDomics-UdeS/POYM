import os

if __name__ == '__main__':
    os.chdir('/')
    from src.models.ranger_forest import HOMRBinaryRGFC
    from src.data.processing.preparing import DataPreparer
    from src.data.processing.datasets import HOMRDataset
    from src.data.processing.constants import *
    from src.data.processing.sampling import KFoldsSampler
    from src.utils.metric_scores import *
    from src.evaluating.evaluation import Evaluator
    from src.models.lstm import HOMRBinaryLSTMC
    from hps.sanity_check_hps import RNN_HPS, RGF_HPS
    from src.utils.delong import compute_median_significant_difference

    task = 'oym'
    # Prepare HOMR learning data
    dp = DataPreparer(task=task)
    df = dp.get__training_cohort

    # Initialize random state
    SEED = 101

    # Initialization of the dictionary containing the evaluation metrics
    evaluation_metrics = [BinaryAccuracy(),
                          BinaryBalancedAccuracy(),
                          BinaryBalancedAccuracy(Reduction.GEO_MEAN),
                          Sensitivity(),
                          Specificity(),
                          AUPRC(),
                          BrierScore(),
                          Precision(),
                          NegativePredictiveValue(),
                          NFN(),
                          NFP(),
                          NTN(),
                          NTP(),
                          F2Score(),
                          F1Score(),
                          BinaryBalancedAccuracy(),
                          AUC()]

    for i, cat_columns in enumerate([CAT_COLS, OTHER_CAT_COL]):
        if i == 0:
            exp_suffix = 'AdmDemoDx'
            print('AdmDemoDx experiments')
        else:
            exp_suffix = 'AdmDemo'
            print('AdmDemo experiments')

        print("### Evaluate using LSTM ###")
        print('### Read the dataset ###')
        # Dataset creation
        dataset = HOMRDataset(df, task, IDS, CONT_COLS, cat_columns, encoding="one hot", temporal=True, to_tensor=True)

        print('### Sampling ###')
        # Nested 5 cross validation sampling
        sampler = KFoldsSampler(dataset=dataset, valid_size=0.1,
                                k=5,
                                inner_k=5)
        masks = sampler(sampling_strategy=-1, temporal_sampling=True, multiple_test_masks=False)

        def update_fixed_params(subset, itr):
            return {'max_epochs': 500,
                    'patience': 50,
                    'num_cont_col': len(subset.cont_cols) + len(subset.cat_cols),
                    'cat_idx': [],
                    'cat_sizes': [],
                    'cat_emb_sizes': []}


        fixed_params = update_fixed_params(dataset, 2)

        evaluator = Evaluator(model_constructor=HOMRBinaryLSTMC,
                              dataset=dataset,
                              masks=masks,
                              hps=RNN_HPS,
                              n_trials=100,
                              evaluation_metrics=evaluation_metrics,
                              fixed_params=fixed_params,
                              fixed_params_update_function=update_fixed_params,
                              evaluation_name=f"LSTM_tuned_{exp_suffix}",
                              save_hps_importance=True,
                              existing_eval=True,
                              save_optimization_history=True,
                              seed=SEED)

        evaluator.evaluate()

        print("### Evaluate using Random Forest ###")
        print('### Read the dataset ###')
        # Dataset creation
        dataset = HOMRDataset(df, task, IDS, CONT_COLS, cat_columns, encoding="one hot")

        # Initialize random state
        SEED = 101

        print('### Sampling ###')
        # Nested 5 cross validation sampling
        sampler = KFoldsSampler(dataset=dataset, valid_size=0.,
                                k=5,
                                inner_k=5)
        masks = sampler(sampling_strategy=-1, cumulate_samples=True, multiple_test_masks=False)

        """
           Evaluator validation with ranger forest
        """

        evaluator = Evaluator(model_constructor=HOMRBinaryRGFC,
                              dataset=dataset,
                              masks=masks,
                              hps=RGF_HPS,
                              n_trials=100,
                              evaluation_metrics=evaluation_metrics,
                              evaluation_name=f"RandomForest_tuned_{exp_suffix}",
                              save_hps_importance=True,
                              save_optimization_history=True,
                              existing_eval=True,
                              seed=SEED)

        evaluator.evaluate()

        print('DeLong test to measure the difference significance')
        compute_median_significant_difference(best_model=f'records_exp/oym/LSTM_tuned_{exp_suffix}',
                                              second_model=f'records_exp/oym/RandomForest_tuned_{exp_suffix}',
                                              section_best_model="test_results",
                                              section_second_model="test_results",
                                              n_splits=5,
                                              temporal_analysis=True,
                                              saving_dir=f'records_exp/p_values/LSTM_vs_RF_{exp_suffix}',
                                              sec='/records.json')
