import pickle


class ModelManager:
    
    @staticmethod
    def save_model(model, name, scaler, features, y_mean, y_std, path='model.pkl'):
        data = {'model': model, 'name': name, 'scaler': scaler,
                'features': features, 'y_mean': y_mean, 'y_std': y_std}
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved: {path}")
    
    @staticmethod
    def load_model(path='model.pkl'):
        with open(path, 'rb') as f:
            return pickle.load(f)

