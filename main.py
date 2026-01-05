from data_processor import DataProcessor
from feature_selector import FeatureSelector
from model_trainer import ModelTrainer
from model_evaluator import ModelEvaluator
from shap_analyzer import SHAPAnalyzer
from model_manager import ModelManager
from delay_analyzer import analyze_delays


DATA_FILE = 'data.csv'
USE_BORUTA = False
USE_DELAY_ANALYSIS = False


if __name__ == '__main__':
    
    processor = DataProcessor()
    X, y, features = processor.load_data(DATA_FILE)
    X_scaled, y_scaled = processor.preprocess(X, y)
    
    if USE_DELAY_ANALYSIS:
        X_scaled, y_scaled, _ = analyze_delays(X_scaled, y_scaled, features)
    
    if USE_BORUTA:
        selector = FeatureSelector()
        X_scaled, features = selector.select_boruta(X_scaled, y_scaled, features)
    
    X_train, X_test, y_train, y_test = processor.split_data(X_scaled, y_scaled)
    
    trainer = ModelTrainer()
    trainer.train_all(X_train, y_train)
    models = trainer.get_models()
    
    evaluator = ModelEvaluator()
    evaluator.evaluate_all(models, X_train, y_train, X_test, y_test)
    evaluator.plot_predictions(models, X_test, y_test)
    evaluator.export_results('model_results.csv')
    
    best_model, best_name = evaluator.get_best_model()
    
    analyzer = SHAPAnalyzer()
    analyzer.analyze(best_model, best_name, X_train, X_test, features)
    
    ModelManager.save_model(best_model, best_name, processor.scaler, 
                           features, processor.y_mean, processor.y_std)

