import pandas as pd

def load_harveststat_data(filepath):
    """Load HarvestStat Africa data"""
    df = pd.read_csv(filepath)
    if 'product' in df.columns:
        df = df[df['product'] == 'Maize'].copy()
    return df

def load_climate_data(filepath):
    """Load climate data"""
    return pd.read_csv(filepath)
