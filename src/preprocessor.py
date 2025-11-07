import pandas as pd
import numpy as np

class DataPreprocessor:
    def __init__(self):
        self.processed_data = None
    
    def merge_datasets(self, harveststat_df, climate_df):
        """Merge HarvestStat with climate data with realistic shock definitions"""
        merged_df = pd.merge(harveststat_df, climate_df, 
                           left_on=['fnid', 'harvest_year'],
                           right_on=['region_id', 'year'], 
                           how='inner')
        
        # REALISTIC SHOCK DEFINITIONS
        # Drought: precipitation less than 200mm (severe drought)
        merged_df['drought_shock'] = (merged_df['precipitation'] < 200).astype(int)
        
        # Conflict: any conflict events (more than 0)
        merged_df['conflict_shock'] = (merged_df['conflict_events'] > 0).astype(int)
        
        # Extreme heat: temperature above 28°C
        merged_df['heat_shock'] = (merged_df['temperature'] > 28).astype(int)
        
        # Combined shock index
        merged_df['total_shocks'] = (merged_df['drought_shock'] + 
                                   merged_df['conflict_shock'] + 
                                   merged_df['heat_shock'])
        
        # Create realistic baseline yields (what yields SHOULD be without shocks)
        # Base yield depends on soil quality and region
        merged_df['baseline_yield'] = merged_df['soil_quality'] * 4.0  # 4.0 tons/ha max potential
        
        # Calculate yield gap (how much was lost due to shocks)
        merged_df['yield_gap'] = merged_df['baseline_yield'] - merged_df['yield']
        
        self.processed_data = merged_df
        return merged_df
    
    def save_processed_data(self, output_path):
        """Save processed data to CSV"""
        if self.processed_data is not None:
            self.processed_data.to_csv(output_path, index=False)
            print(f"Data saved to {output_path}")
