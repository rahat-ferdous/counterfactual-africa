import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.data_loader import load_harveststat_data, load_climate_data
from src.preprocessor import DataPreprocessor
from src.model import CounterfactualModel

# Page configuration
st.set_page_config(
    page_title="Counterfactual Africa",
    page_icon="🌍",
    layout="wide"
)

# Title and description
st.title("🌍 Counterfactual Africa")
st.markdown("""
### Estimate the Real Impact of Climate and Conflict Shocks
This tool uses machine learning to answer **"what-if"** questions about agricultural productivity across Africa.
""")

# Load data
@st.cache_data
def load_data():
    hs_data = load_harveststat_data('data/raw/sample_harveststat_data.csv')
    climate_data = load_climate_data('data/raw/sample_climate_data.csv')
    
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.merge_datasets(hs_data, climate_data)
    return processed_data

@st.cache_resource
def train_model(data):
    model = CounterfactualModel()
    model.train(data)
    return model

# Load data
data = load_data()

# Sidebar for controls
st.sidebar.header("🔧 Analysis Controls")

# Intervention type
intervention_type = st.sidebar.selectbox(
    "What-if Scenario",
    ["Remove Drought", "Remove Conflict", "Remove Heat Stress", "Remove All Shocks"]
)

# Region selection
target_region = st.sidebar.selectbox("Select Region", data['fnid'].unique())

# Year selection
target_year = st.sidebar.selectbox("Select Year", sorted(data['harvest_year'].unique()))

# Intervention mapping
intervention_map = {
    "Remove Drought": {'drought_shock': 0},
    "Remove Conflict": {'conflict_shock': 0},
    "Remove Heat Stress": {'heat_shock': 0},
    "Remove All Shocks": {'drought_shock': 0, 'conflict_shock': 0, 'heat_shock': 0}
}

# Main analysis
st.header("📊 Regional Analysis")

if st.sidebar.button("Run Counterfactual Analysis", type="primary"):
    with st.spinner("Training model and analyzing scenarios..."):
        # Train model
        model = train_model(data)
        
        # Get target data
        target_data = data[
            (data['fnid'] == target_region) & 
            (data['harvest_year'] == target_year)
        ].copy()
        
        if len(target_data) > 0:
            # Get actual values
            actual_yield = target_data['yield'].values[0]
            precipitation = target_data['precipitation'].values[0]
            temperature = target_data['temperature'].values[0]
            conflict_events = target_data['conflict_events'].values[0]
            
            # Predict counterfactual
            intervention = intervention_map[intervention_type]
            counterfactual_yield = model.predict_counterfactual(target_data, intervention)[0]
            
            # Calculate improvement
            improvement = counterfactual_yield - actual_yield
            improvement_pct = (improvement / actual_yield) * 100
            
            # Display results
            st.success("✅ Analysis Complete!")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Actual Yield",
                    value=f"{actual_yield:.2f} t/ha",
                    delta=None
                )
            
            with col2:
                st.metric(
                    label="Counterfactual Yield",
                    value=f"{counterfactual_yield:.2f} t/ha",
                    delta=f"+{improvement:.2f} t/ha"
                )
            
            with col3:
                st.metric(
                    label="Improvement",
                    value=f"+{improvement_pct:.1f}%",
                    delta=None
                )
                
            with col4:
                st.metric(
                    label="Potential Gain",
                    value=f"{improvement * target_data['area_hectares'].values[0]:.0f} tons",
                    delta=None
                )
            
            # Conditions analysis
            st.subheader("🌦️ Environmental Conditions")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                status = "⚠️ Drought" if target_data['drought_shock'].values[0] == 1 else "✅ Normal"
                st.write(f"**Rainfall:** {precipitation}mm ({status})")
            
            with col2:
                status = "⚠️ Heat Stress" if target_data['heat_shock'].values[0] == 1 else "✅ Normal"
                st.write(f"**Temperature:** {temperature}°C ({status})")
            
            with col3:
                status = "⚠️ Conflict" if conflict_events > 0 else "✅ Peaceful"
                st.write(f"**Conflict Events:** {conflict_events} ({status})")
            
            # Visualization
            st.subheader("📈 Impact Visualization")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Yield comparison
            scenarios = ['Actual', 'Counterfactual']
            yields = [actual_yield, counterfactual_yield]
            colors = ['#ff6b6b', '#51cf66']
            
            bars1 = ax1.bar(scenarios, yields, color=colors, alpha=0.8)
            ax1.set_ylabel('Yield (tons/hectare)')
            ax1.set_title(f'Yield Comparison: {target_region} ({target_year})')
            ax1.grid(True, alpha=0.3)
            
            for bar, yield_val in zip(bars1, yields):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{yield_val:.2f}', ha='center', va='bottom', fontweight='bold')
            
            # Shocks analysis
            shocks_present = [
                target_data['drought_shock'].values[0],
                target_data['conflict_shock'].values[0], 
                target_data['heat_shock'].values[0]
            ]
            shocks_removed = [0, 0, 0]  # All removed in counterfactual
            
            shock_types = ['Drought', 'Conflict', 'Heat Stress']
            x_pos = np.arange(len(shock_types))
            
            bars2 = ax2.bar(x_pos - 0.2, shocks_present, 0.4, label='Actual', color='#ff6b6b', alpha=0.8)
            bars3 = ax2.bar(x_pos + 0.2, shocks_removed, 0.4, label='Counterfactual', color='#51cf66', alpha=0.8)
            
            ax2.set_xlabel('Shock Type')
            ax2.set_ylabel('Shock Presence (0=No, 1=Yes)')
            ax2.set_title('Shocks: Actual vs Counterfactual')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(shock_types)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Interpretation
            st.subheader("💡 Policy Implications")
            
            if improvement > 0:
                st.info(f"""
                **Key Finding:** Removing these shocks could have increased maize production by **{improvement_pct:.1f}%** in {target_region} during {target_year}.
                
                **Potential impact:**
                - Additional **{improvement:.2f} tons per hectare**
                - Total regional gain: **{improvement * target_data['area_hectares'].values[0]:.0f} tons** of maize
                - Enough to feed approximately **{int((improvement * target_data['area_hectares'].values[0]) / 0.15)} people** for a year
                """)
            else:
                st.warning("""
                **Analysis Note:** The model suggests limited impact from shock removal in this scenario.
                This could indicate other limiting factors (soil quality, management practices, etc.)
                are more significant constraints in this region and year.
                """)
            
        else:
            st.error("❌ No data found for the selected region and year.")

# Regional trends
st.header("📅 Historical Trends")
region_data = data[data['fnid'] == target_region]

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(region_data['harvest_year'], region_data['yield'], marker='o', linewidth=2, label='Actual Yield')
ax.axhline(y=region_data['baseline_yield'].mean(), color='green', linestyle='--', alpha=0.7, label='Potential Yield')
ax.set_xlabel('Year')
ax.set_ylabel('Yield (tons/hectare)')
ax.set_title(f'Maize Yield Trend: {target_region}')
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# Data summary
with st.expander("📋 View Raw Data"):
    st.dataframe(data[data['fnid'] == target_region], use_container_width=True)
