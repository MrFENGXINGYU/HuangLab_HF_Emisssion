from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy
from config import RANDOM_STATE, BORUTA_MAX_ITER, BORUTA_RF_PARAMS


class FeatureSelector:
    
    def __init__(self):
        self.selected_features = None
        
    def select_boruta(self, X, y, feature_names):
        rf = RandomForestRegressor(**BORUTA_RF_PARAMS, random_state=RANDOM_STATE)
        boruta = BorutaPy(rf, n_estimators='auto', max_iter=BORUTA_MAX_ITER, 
                         alpha=0.05, two_step=False, random_state=RANDOM_STATE)
        boruta.fit(X, y)
        
        mask = boruta.support_
        self.selected_features = [feature_names[i] for i in range(len(feature_names)) if mask[i]]
        X_selected = X[:, mask]
        
        print(f"Selected {len(self.selected_features)}/{len(feature_names)} features")
        return X_selected, self.selected_features

