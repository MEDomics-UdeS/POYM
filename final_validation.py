if __name__ == '__main__':
    from src.data.processing.preparing import DataPreparer
    from src.data.processing.datasets import HOMRDataset
    from src.data.processing.constants import *
    from src.data.processing.sampling import SimpleSampler
    from src.utils.metric_scores import *
    from src.evaluating.evaluation import Evaluator
    from src.models.lstm import HOMRBinaryLSTMC
    from hps.sanity_check_hps import RNN_HPS
    import numpy as np

    task = 'oym'
    # Prepare HOMR learning data
    dp = DataPreparer(task=task, test_fie='csvs/df_holdout.csv')
    df_train = dp.get__training_cohort
    df_test = dp.get__testing_cohort
    df_test = dp.create_CDSS_eligble_cohort(df_test)
    df = pd.concat([df_train, df_test]).reset_index(drop=True)

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
        # Inner nested 5 cross validation sampling
        sampler = SimpleSampler(dataset=dataset, n_inner=5, valid_size=0.1)
        masks = sampler(learning_ids=np.array(df_train[IDS].unique()),
                        test_ids=np.array(df_test[IDS].unique()),
                        sampling_strategy=-1,
                        temporal_sampling=True,
                        multiple_test_masks=False)

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
                              n_trials=3,
                              evaluation_metrics=evaluation_metrics,
                              fixed_params=fixed_params,
                              fixed_params_update_function=update_fixed_params,
                              evaluation_name=f"LSTM_tuned_holdout_{exp_suffix}",
                              save_hps_importance=True,
                              existing_eval=True,
                              save_optimization_history=True,
                              seed=SEED)

        evaluator.evaluate()
