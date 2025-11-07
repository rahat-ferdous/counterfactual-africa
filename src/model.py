import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

class CounterfactualModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.is_trained = False
    
    def prepare_features(self, df):
        """Prepare features for training"""
        feature_columns = ['precipitation', 'temperature', 'drought_shock', 
                          'conflict_shock', 'harvest_year']
        
        # Select features and handle missing values
        X = df[feature_columns].fillna(method='ffill')
        y = df['yield']
        
        return X, y
    
    def train(self, df, save_path=None):
        """Train the counterfactual model"""
        print("Training counterfactual model...")
        
        X, y = self.prepare_features(df)
        self.model.fit(X, y)
        self.is_trained = True
        
        if save_path:
            joblib.dump(self.model, save_path)
            print(f"Model saved to {save_path}")
    
    def predict_counterfactual(self, df, intervention):
        """
        Predict counterfactual outcomes
        intervention: dict e.g., {'drought_shock': 0, 'conflict_shock': 0}
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X, _ = self.prepare_features(df)
        
        # Apply intervention
        X_counterfactual = X.copy()
        for feature, value in intervention.items():
            if feature in X_counterfactual.columns:
                X_counterfactual[feature] = value
        
        predictions = self.model.predict(X_counterfactual)
        return predictions
