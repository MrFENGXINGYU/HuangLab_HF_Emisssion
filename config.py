import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
N_ITER_BAYESIAN = 50
BORUTA_MAX_ITER = 100
SHAP_SAMPLE_SIZE = 1000

XGBOOST_PARAMS = {
    'n_estimators': (10, 600),
    'max_depth': (3, 120),
    'learning_rate': (0.1, 1.0),
    'subsample': (0.5, 1.0),
    'gamma': (0.0, 0.5)
}

GBR_PARAMS = {
    'n_estimators': (10, 600),
    'max_depth': (3, 15),
    'max_features': (1, 10)
}

CATBOOST_PARAMS = {
    'iterations': (10, 600),
    'depth': (3, 15),
    'learning_rate': (0.03, 1.0)
}

RF_PARAMS = {
    'n_estimators': (10, 600),
    'max_depth': (3, 20),
    'max_features': (1, 10),
    'min_samples_leaf': (1, 10)
}

BORUTA_RF_PARAMS = {
    'n_estimators': 500,
    'max_depth': 15,
    'n_jobs': -1
}

