from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from config import (RANDOM_STATE, CV_FOLDS, N_ITER_BAYESIAN,
                   XGBOOST_PARAMS, GBR_PARAMS, CATBOOST_PARAMS, RF_PARAMS)


class ModelTrainer:
    
    def __init__(self):
        self.models = {}
        
    def _get_spaces(self):
        return {
            'XGBoost': {
                'model': XGBRegressor(random_state=RANDOM_STATE),
                'params': {
                    'n_estimators': Integer(*XGBOOST_PARAMS['n_estimators']),
                    'max_depth': Integer(*XGBOOST_PARAMS['max_depth']),
                    'learning_rate': Real(*XGBOOST_PARAMS['learning_rate'], prior='log-uniform'),
                    'subsample': Real(*XGBOOST_PARAMS['subsample']),
                    'gamma': Real(*XGBOOST_PARAMS['gamma'])
                }
            },
            'GBR': {
                'model': GradientBoostingRegressor(random_state=RANDOM_STATE),
                'params': {
                    'n_estimators': Integer(*GBR_PARAMS['n_estimators']),
                    'max_depth': Integer(*GBR_PARAMS['max_depth']),
                    'max_features': Integer(*GBR_PARAMS['max_features'])
                }
            },
            'CatBoost': {
                'model': CatBoostRegressor(random_state=RANDOM_STATE, verbose=0),
                'params': {
                    'iterations': Integer(*CATBOOST_PARAMS['iterations']),
                    'depth': Integer(*CATBOOST_PARAMS['depth']),
                    'learning_rate': Real(*CATBOOST_PARAMS['learning_rate'], prior='log-uniform')
                }
            },
            'RandomForest': {
                'model': RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
                'params': {
                    'n_estimators': Integer(*RF_PARAMS['n_estimators']),
                    'max_depth': Integer(*RF_PARAMS['max_depth']),
                    'max_features': Integer(*RF_PARAMS['max_features']),
                    'min_samples_leaf': Integer(*RF_PARAMS['min_samples_leaf'])
                }
            }
        }
    
    def train_all(self, X_train, y_train):
        spaces = self._get_spaces()
        
        for name, cfg in spaces.items():
            print(f"\nTraining {name}...")
            
            opt = BayesSearchCV(cfg['model'], cfg['params'], n_iter=N_ITER_BAYESIAN,
                               cv=CV_FOLDS, n_jobs=-1, random_state=RANDOM_STATE,
                               scoring='neg_mean_squared_error')
            opt.fit(X_train, y_train)
            self.models[name] = opt.best_estimator_
            print(f"{name} CV: {-opt.best_score_:.4f}")
    
    def get_models(self):
        return self.models

