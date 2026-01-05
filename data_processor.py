import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from config import RANDOM_STATE, TEST_SIZE


class DataProcessor:
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.y_mean = None
        self.y_std = None
        
    def load_data(self, filepath):
        df = pd.read_csv(filepath)
        
        if df.isnull().any().any():
            df = df.interpolate(method='linear', limit_direction='both')
        
        y = df.iloc[:, -1].values
        X = df.iloc[:, :-1].values
        feature_names = df.columns[:-1].tolist()
        print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features")
        return X, y, feature_names
    
    def preprocess(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.y_mean = np.mean(y)
        self.y_std = np.std(y)
        y_scaled = (y - self.y_mean) / self.y_std
        return X_scaled, y_scaled
    
    def split_data(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
        return X_train, X_test, y_train, y_test

