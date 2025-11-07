import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

class CounterfactualModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
    
    def prepare_features(self, df):
        """Prepare features for training"""
        features = ['precipitation', 'temperature', 'drought_shock', 'conflict_shock', 'harvest_year']
        X = df[features]
        y = df['yield']
        return X, y
    
    def train(self, df, save_path=None):
        """Train the model"""
        print("Training model...")
        X, y = self.prepare_features(df)
        self.model.fit(X, y)
        self.is_trained = True
        if save_path:
            joblib.dump(self.model, save_path)
            print(f"Model saved to {save_path}")
    
    def predict_counterfactual(self, df, intervention):
        """Predict what would happen under different conditions"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        X, _ = self.prepare_features(df)
        X_counterfactual = X.copy()
        for feature, value in intervention.items():
            if feature in X_counterfactual.columns:
                X_counterfactual[feature] = value
        
        return self.model.predict(X_counterfactual)
