import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import load_harveststat_data, load_climate_data
from src.preprocessor import DataPreprocessor

def main():
    print("Loading data...")
    hs_data = load_harveststat_data('data/raw/sample_harveststat_data.csv')
    climate_data = load_climate_data('data/raw/sample_climate_data.csv')
    
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.merge_datasets(hs_data, climate_data)
    preprocessor.save_processed_data('data/processed/training_data.csv')
    print("Data processing complete!")

if __name__ == "__main__":
    main()
