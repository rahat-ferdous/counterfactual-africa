import pandas as pd

class DataPreprocessor:
    def __init__(self):
        self.processed_data = None
    
    def merge_datasets(self, harveststat_df, climate_df):
        """Merge HarvestStat with climate data"""
        merged_df = pd.merge(harveststat_df, climate_df, 
                           left_on=['fnid', 'harvest_year'],
                           right_on=['region_id', 'year'], 
                           how='inner')
        
        merged_df['drought_shock'] = (merged_df['precipitation'] < 100).astype(int)
        merged_df['conflict_shock'] = (merged_df['conflict_events'] > 0).astype(int)
        
        self.processed_data = merged_df
        return merged_df
    
    def save_processed_data(self, output_path):
        """Save processed data to CSV"""
        if self.processed_data is not None:
            self.processed_data.to_csv(output_path, index=False)
            print(f"Data saved to {output_path}")
