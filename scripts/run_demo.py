import sys
import os
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model import CounterfactualModel

def main():
    data = pd.read_csv('data/processed/training_data.csv')
    model = CounterfactualModel()
    model.train(data, save_path='outputs/models/counterfactual_model.pkl')
    
    test_cases = data[data['drought_shock'] == 1].head(3)
    counterfactual_yields = model.predict_counterfactual(test_cases, {'drought_shock': 0})
    
    print("\n=== COUNTERFACTUAL RESULTS ===")
    for i, (real, cf) in enumerate(zip(test_cases['yield'].values, counterfactual_yields)):
        improvement = cf - real
        print(f"Region {i+1}: {real:.2f} → {cf:.2f} (Improvement: +{improvement:.2f})")

if __name__ == "__main__":
    main()
