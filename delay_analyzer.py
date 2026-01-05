import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class DelayAnalyzer:
    
    def __init__(self, max_delay_minutes=20, sampling_interval_seconds=5):
        self.max_delay = max_delay_minutes
        self.interval = sampling_interval_seconds
        self.max_steps = int(max_delay_minutes * 60 / sampling_interval_seconds)
        self.optimal_delays = {}
        self.mi_scores = {}
        
    def calculate_mi_for_delay(self, X_feature, y_target, delay_steps):
        if delay_steps == 0:
            return mutual_info_regression(X_feature.reshape(-1, 1), y_target, random_state=42)[0]
        
        X_delayed = X_feature[:-delay_steps]
        y_aligned = y_target[delay_steps:]
        
        if len(X_delayed) < 10:
            return 0
        
        return mutual_info_regression(X_delayed.reshape(-1, 1), y_aligned, random_state=42)[0]
    
    def find_optimal_delay(self, X, y, feature_names):
        n_features = X.shape[1]
        
        for i in range(n_features):
            mi_values = []
            
            for delay in range(self.max_steps + 1):
                mi = self.calculate_mi_for_delay(X[:, i], y, delay)
                mi_values.append(mi)
            
            optimal_idx = np.argmax(mi_values)
            optimal_delay_minutes = (optimal_idx * self.interval) / 60.0
            
            self.optimal_delays[feature_names[i]] = {
                'delay_steps': optimal_idx,
                'delay_minutes': optimal_delay_minutes,
                'max_mi': mi_values[optimal_idx]
            }
            self.mi_scores[feature_names[i]] = mi_values
        
        return self.optimal_delays
    
    def apply_optimal_delays(self, X, feature_names):
        max_delay_step = max([d['delay_steps'] for d in self.optimal_delays.values()])
        
        if max_delay_step == 0:
            return X
        
        X_delayed = np.zeros((X.shape[0] - max_delay_step, X.shape[1]))
        
        for i, fname in enumerate(feature_names):
            delay = self.optimal_delays[fname]['delay_steps']
            shift = max_delay_step - delay
            
            if shift == 0:
                X_delayed[:, i] = X[max_delay_step:, i]
            else:
                X_delayed[:, i] = X[shift:-max_delay_step+shift, i]
        
        return X_delayed
    
    def plot_mi_curves(self, top_n=10, save_path='delay_mi.png'):
        sorted_features = sorted(self.optimal_delays.items(), 
                                key=lambda x: x[1]['max_mi'], reverse=True)
        
        fig, axes = plt.subplots(2, 5, figsize=(18, 8))
        axes = axes.ravel()
        
        for idx, (fname, info) in enumerate(sorted_features[:top_n]):
            mi_vals = self.mi_scores[fname]
            time_mins = np.arange(len(mi_vals)) * self.interval / 60.0
            
            axes[idx].plot(time_mins, mi_vals, 'b-', linewidth=1.5)
            axes[idx].axvline(info['delay_minutes'], color='r', linestyle='--', 
                            label=f"Optimal: {info['delay_minutes']:.1f} min")
            axes[idx].set_xlabel('Delay (min)', fontsize=9)
            axes[idx].set_ylabel('MI', fontsize=9)
            axes[idx].set_title(f"{fname}", fontsize=10)
            axes[idx].legend(fontsize=7)
            axes[idx].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def export_delays(self, filepath='optimal_delays.csv'):
        data = []
        for fname, info in self.optimal_delays.items():
            data.append({
                'Feature': fname,
                'Delay_Minutes': info['delay_minutes'],
                'Delay_Steps': info['delay_steps'],
                'Max_MI': info['max_mi']
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('Max_MI', ascending=False)
        df.to_csv(filepath, index=False)
        
        print(f"\nTop 10 features by MI:")
        print(df.head(10).to_string(index=False))
        
        return df


def analyze_delays(X, y, feature_names, max_delay_min=20, interval_sec=5):
    analyzer = DelayAnalyzer(max_delay_min, interval_sec)
    
    print(f"Analyzing delays (0-{max_delay_min} min)...")
    optimal_delays = analyzer.find_optimal_delay(X, y, feature_names)
    
    avg_delay = np.mean([d['delay_minutes'] for d in optimal_delays.values()])
    print(f"Average optimal delay: {avg_delay:.2f} minutes")
    
    analyzer.plot_mi_curves(top_n=10, save_path='delay_mi.png')
    delay_df = analyzer.export_delays('optimal_delays.csv')
    
    X_delayed = analyzer.apply_optimal_delays(X, feature_names)
    y_aligned = y[-X_delayed.shape[0]:]
    
    print(f"Applied delays: {X.shape[0]} -> {X_delayed.shape[0]} samples")
    
    return X_delayed, y_aligned, delay_df

