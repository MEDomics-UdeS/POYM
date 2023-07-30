"""
File used to store hyperparameters used for sanity checks
"""

from src.models.xgboost_ import XGBoostHP
from src.models.ranger_forest import RangerForestHP
from src.models.lstm import RNNHP
from src.utils.hyperparameters import Range

XGBOOST_HPS = {
    XGBoostHP.ALPHA.name: {
        Range.MIN: 0,
        Range.MAX: 5
    },
    XGBoostHP.BETA.name: {
        Range.MIN: 0,
        Range.MAX: 5
    },
    XGBoostHP.LR.name: {
        Range.MIN: 0.001,
        Range.MAX: 0.5
    },
    XGBoostHP.MAX_DEPTH.name: {
        Range.MIN: 4,
        Range.MAX: 12,
    },
    XGBoostHP.SUBSAMPLE.name: {
        Range.VALUE: 1,
    },
    XGBoostHP.WEIGHT.name: {
        Range.MIN: 0.1,
        Range.MAX: 0.9,
    },
    XGBoostHP.N_ESTIMATORS.name: {
        Range.MIN: 100,
        Range.MAX: 500
    }
}

RNN_HPS = {
    RNNHP.ALPHA.name: {
        Range.VALUE: 0,
    },
    RNNHP.BATCH_SIZE.name: {
        Range.VALUE: 100
    },
    RNNHP.BETA.name: {
        Range.MIN: 0,
        Range.MAX: 0.0001
    },
    RNNHP.LR.name: {
        Range.MIN: 0.00001,
        Range.MAX: 0.001
    },
    RNNHP.N_LAYER.name: {
        Range.VALUE: 1
    },
    RNNHP.N_UNIT.name: {
        Range.MIN: 16,
        Range.MAX: 64,
        Range.STEP: 16
    },
    RNNHP.WEIGHT.name: {
        Range.MIN: 0.1,
        Range.MAX: 0.9,
        Range.STEP: 0.1
    },
    RNNHP.BIDIRECTIONAL.name: {
        Range.VALUE: True,
    },
    RNNHP.DROPOUT.name: {
        Range.VALUE: 0
    },
}

RGF_HPS = {
    RangerForestHP.MTRY.name: {
        Range.MIN: 10,
        Range.MAX: 20,
        Range.STEP: 5
    },
    RangerForestHP.MIN_NODE_SIZE.name: {
        Range.MIN: 10,
        Range.MAX: 80,
        Range.STEP: 10
    },
    RangerForestHP.MAX_DEPTH.name: {
        Range.VALUE: 0,
    },
    RangerForestHP.N_ESTIMATORS.name: {
        Range.MIN: 128,
        Range.MAX: 1024,
        Range.STEP: 128,
    },
    RangerForestHP.WEIGHT.name: {
        Range.MIN: 0.1,
        Range.MAX: 0.9,
    },
}
