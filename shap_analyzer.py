import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from config import SHAP_SAMPLE_SIZE


class SHAPAnalyzer:
    
    def __init__(self):
        self.shap_values = None
        self.explainer = None
        self.X_sample = None
        
    def analyze(self, model, model_name, X_train, X_test, feature_names, save_dir='shap'):
        os.makedirs(save_dir, exist_ok=True)
        print(f"\nSHAP analysis for {model_name}...")
        
        self.explainer = shap.TreeExplainer(model)
        n = min(SHAP_SAMPLE_SIZE, X_test.shape[0])
        self.X_sample = X_test[:n]
        self.shap_values = self.explainer.shap_values(self.X_sample)
        
        self._plot_summary(feature_names, save_dir)
        self._plot_feature_importance(feature_names, save_dir)
        feat_imp = self._calculate_feature_importance(feature_names, save_dir)
        self._plot_dependence(feature_names, feat_imp, save_dir)
        self._plot_force(feature_names, save_dir)
        self._plot_waterfall(feature_names, save_dir)
        
        print("SHAP done")
        return feat_imp
    
    def _plot_summary(self, features, save_dir):
        plt.figure(figsize=(11, 9))
        shap.summary_plot(self.shap_values, self.X_sample, 
                         feature_names=features, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/summary.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_importance(self, features, save_dir):
        plt.figure(figsize=(11, 9))
        shap.summary_plot(self.shap_values, self.X_sample,
                         feature_names=features, plot_type='bar', 
                         show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _calculate_feature_importance(self, features, save_dir):
        shap_mean = np.abs(self.shap_values).mean(axis=0)
        corrs = [np.corrcoef(self.X_sample[:, i], self.shap_values[:, i])[0, 1] 
                for i in range(self.X_sample.shape[1])]
        
        df = pd.DataFrame({
            'Feature': features,
            'Mean_SHAP': shap_mean,
            'Percentage': shap_mean / shap_mean.sum() * 100,
            'Correlation': corrs
        }).sort_values('Mean_SHAP', ascending=False)
        
        df.to_csv(f'{save_dir}/importance.csv', index=False)
        print(f"\nTop features:\n{df.head(10).to_string(index=False)}")
        return df
    
    def _plot_dependence(self, features, feat_df, save_dir):
        top_idx = feat_df.head(5).index.tolist()
        fig, axes = plt.subplots(2, 3, figsize=(17, 11))
        axes = axes.ravel()
        
        for i, idx in enumerate(top_idx[:5]):
            shap.dependence_plot(idx, self.shap_values, self.X_sample,
                               feature_names=features, ax=axes[i], show=False)
        fig.delaxes(axes[5])
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/dependence.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_force(self, features, save_dir):
        shap.initjs()
        shap.force_plot(self.explainer.expected_value, self.shap_values[0, :],
                       self.X_sample[0, :], feature_names=features,
                       matplotlib=True, show=False)
        plt.savefig(f'{save_dir}/force.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_waterfall(self, features, save_dir):
        plt.figure(figsize=(11, 9))
        shap.waterfall_plot(
            shap.Explanation(values=self.shap_values[0],
                           base_values=self.explainer.expected_value,
                           data=self.X_sample[0],
                           feature_names=features),
            max_display=20, show=False
        )
        plt.tight_layout()
        plt.savefig(f'{save_dir}/waterfall.png', dpi=300, bbox_inches='tight')
        plt.close()

