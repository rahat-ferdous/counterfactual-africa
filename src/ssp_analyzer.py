import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

class SSPAnalyzer:
    def __init__(self, base_model):
        self.base_model = base_model
        self.ssp_scenarios = None
        self.results = {}
    
    def load_ssp_scenarios(self, filepath='data/raw/ssp_scenarios.csv'):
        """Load SSP scenario definitions"""
        self.ssp_scenarios = pd.read_csv(filepath)
        return self.ssp_scenarios
    
    def create_future_baseline(self, region_data, target_year, ssp_scenario):
        """Create future baseline conditions for a given SSP scenario"""
        
        # Get historical baseline (average of last 5 years)
        recent_data = region_data[region_data['harvest_year'] >= region_data['harvest_year'].max() - 5]
        baseline = recent_data.mean()
        
        # Get SSP parameters for target year
        ssp_params = self.ssp_scenarios[
            (self.ssp_scenarios['ssp_scenario'] == ssp_scenario) & 
            (self.ssp_scenarios['year'] == target_year)
        ].iloc[0]
        
        # Create future baseline with SSP adjustments
        future_baseline = {
            'precipitation': baseline['precipitation'] * ssp_params['precipitation_change'],
            'temperature': baseline['temperature'] + ssp_params['temperature_change'],
            'soil_quality': baseline['soil_quality'] * ssp_params['agricultural_investment'],
            'harvest_year': target_year,
            # Shock probabilities based on SSP narrative
            'drought_shock_prob': 0.3 * (2 - ssp_params['precipitation_change']),  # Higher if drier
            'conflict_shock_prob': ssp_params['conflict_probability'],
            'heat_shock_prob': min(0.8, ssp_params['temperature_change'] / 5.0)  # Higher if hotter
        }
        
        return future_baseline, ssp_params
    
    def generate_future_scenarios(self, region_data, region_id, target_years=None):
        """Generate future scenarios for all SSPs"""
        
        if target_years is None:
            target_years = [2025, 2030, 2035, 2040, 2045, 2050]
        
        region_data = region_data[region_data['fnid'] == region_id].copy()
        scenarios = {}
        
        for ssp in ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']:
            scenarios[ssp] = {}
            
            for year in target_years:
                # Create baseline for this SSP-year combination
                baseline, ssp_params = self.create_future_baseline(region_data, year, ssp)
                
                # Generate multiple realizations with random shocks
                realizations = []
                for realization in range(100):  # Monte Carlo simulation
                    scenario_data = baseline.copy()
                    
                    # Apply random shocks based on probabilities
                    scenario_data['drought_shock'] = 1 if np.random.random() < baseline['drought_shock_prob'] else 0
                    scenario_data['conflict_shock'] = 1 if np.random.random() < baseline['conflict_shock_prob'] else 0
                    scenario_data['heat_shock'] = 1 if np.random.random() < baseline['heat_shock_prob'] else 0
                    
                    # Adjust precipitation if drought occurs
                    if scenario_data['drought_shock'] == 1:
                        scenario_data['precipitation'] *= np.random.uniform(0.3, 0.6)
                    
                    # Predict yield for this realization
                    scenario_df = pd.DataFrame([scenario_data])
                    predicted_yield = self.base_model.predict_counterfactual(
                        scenario_df, {})[0]
                    
                    realizations.append({
                        'year': year,
                        'ssp_scenario': ssp,
                        'realization': realization,
                        'predicted_yield': predicted_yield,
                        'drought_shock': scenario_data['drought_shock'],
                        'conflict_shock': scenario_data['conflict_shock'],
                        'heat_shock': scenario_data['heat_shock'],
                        'precipitation': scenario_data['precipitation'],
                        'temperature': scenario_data['temperature']
                    })
                
                scenarios[ssp][year] = pd.DataFrame(realizations)
        
        self.results[region_id] = scenarios
        return scenarios
    
    def calculate_ssp_impacts(self, region_id, base_year=2020):
        """Calculate impacts relative to baseline year"""
        
        if region_id not in self.results:
            raise ValueError(f"No results for region {region_id}")
        
        impacts = {}
        scenarios = self.results[region_id]
        
        # Get baseline yield (average of recent years)
        baseline_yield = self._get_historical_baseline(region_id, base_year)
        
        for ssp, years_data in scenarios.items():
            impacts[ssp] = {}
            
            for year, realizations in years_data.items():
                avg_yield = realizations['predicted_yield'].mean()
                yield_change = ((avg_yield - baseline_yield) / baseline_yield) * 100
                
                # Risk metrics
                risk_metrics = {
                    'mean_yield': avg_yield,
                    'yield_change_pct': yield_change,
                    'drought_risk': realizations['drought_shock'].mean(),
                    'conflict_risk': realizations['conflict_shock'].mean(),
                    'heat_risk': realizations['heat_shock'].mean(),
                    'yield_variance': realizations['predicted_yield'].var(),
                    'probability_20pct_decline': (realizations['predicted_yield'] < baseline_yield * 0.8).mean(),
                    'probability_40pct_decline': (realizations['predicted_yield'] < baseline_yield * 0.6).mean()
                }
                
                impacts[ssp][year] = risk_metrics
        
        return impacts
    
    def _get_historical_baseline(self, region_id, base_year=2020):
        """Get historical baseline yield for impact calculation"""
        # This would use your actual historical data
        # For now, return a reasonable baseline
        return 2.5  # tons/ha - you can make this data-driven
