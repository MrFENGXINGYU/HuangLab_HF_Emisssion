import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
from config import RANDOM_STATE, CV_FOLDS


class ModelEvaluator:
    
    def __init__(self):
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def evaluate_all(self, models, X_train, y_train, X_test, y_test):
        print("\nEvaluating models...")
        
        for name, model in models.items():
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            kfold = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
            cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='mae')
            
            self.results[name] = {
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                'train_mae': mean_absolute_error(y_train, y_train_pred),
                'train_r2': r2_score(y_train, y_train_pred),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                'test_mae': mean_absolute_error(y_test, y_test_pred),
                'test_r2': r2_score(y_test, y_test_pred),
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std(),
                'y_test_pred': y_test_pred
            }
            
            r = self.results[name]
            print(f"{name}: Test R²={r['test_r2']:.3f}, RMSE={r['test_rmse']:.3f}, MAE={r['test_mae']:.3f}")
        
        best_name = max(self.results.keys(), key=lambda k: self.results[k]['test_r2'])
        self.best_model = models[best_name]
        self.best_model_name = best_name
        print(f"\nBest: {best_name}")
        
        return self.results
    
    def plot_predictions(self, models, X_test, y_test, save_path='predictions.png'):
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        axes = axes.ravel()
        
        for idx, (name, model) in enumerate(models.items()):
            y_pred = model.predict(X_test)
            r = self.results[name]
            
            axes[idx].scatter(y_test, y_pred, alpha=0.4, s=8)
            lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
            axes[idx].plot(lims, lims, 'r--', lw=2)
            axes[idx].set_xlabel('Actual')
            axes[idx].set_ylabel('Predicted')
            axes[idx].set_title(f'{name}\nR²={r["test_r2"]:.3f}, RMSE={r["test_rmse"]:.3f}')
            axes[idx].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def get_best_model(self):
        return self.best_model, self.best_model_name
    
    def export_results(self, filepath='results.csv'):
        df = pd.DataFrame(self.results).T
        df = df.drop('y_test_pred', axis=1)
        df.to_csv(filepath)
        print(f"Saved: {filepath}")

