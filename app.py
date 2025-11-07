import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
### Estimate the Impact of Climate Shocks on African Agriculture
This interactive tool demonstrates how machine learning can answer **"what-if"** questions about agricultural productivity.
""")

# Sidebar for controls
st.sidebar.header("Analysis Controls")

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

# Load and display data
st.header("📊 Data Overview")
data = load_data()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Sample Data")
    st.dataframe(data.head(10), use_container_width=True)

with col2:
    st.subheader("Data Summary")
    st.write(f"**Total records:** {len(data)}")
    st.write(f"**Regions:** {data['fnid'].nunique()}")
    st.write(f"**Years:** {data['harvest_year'].nunique()}")
    st.write(f"**Average yield:** {data['yield'].mean():.2f} tons/ha")

# Visualization
st.header("📈 Yield Trends")
fig, ax = plt.subplots(figsize=(10, 6))
for region in data['fnid'].unique():
    region_data = data[data['fnid'] == region]
    ax.plot(region_data['harvest_year'], region_data['yield'], 
            label=region, marker='o')
ax.set_xlabel("Year")
ax.set_ylabel("Yield (tons/ha)")
ax.set_title("Maize Yield Trends by Region")
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# Counterfactual Analysis
st.header("🔮 Counterfactual Analysis")
st.markdown("""
### What would yields have been without shocks?
Select a scenario to simulate:
""")

col1, col2, col3 = st.columns(3)

with col1:
    intervention_type = st.selectbox(
        "Intervention Type",
        ["Remove Drought", "Remove Conflict", "Remove Both"]
    )

with col2:
    target_region = st.selectbox(
        "Target Region",
        data['fnid'].unique()
    )

with col3:
    target_year = st.selectbox(
        "Target Year", 
        data['harvest_year'].unique()
    )

# Map intervention type to parameters
intervention_map = {
    "Remove Drought": {'drought_shock': 0},
    "Remove Conflict": {'conflict_shock': 0},
    "Remove Both": {'drought_shock': 0, 'conflict_shock': 0}
}

if st.button("Run Counterfactual Analysis", type="primary"):
    with st.spinner("Training model and running analysis..."):
        # Train model
        model = train_model(data)
        
        # Get the specific case to analyze
        target_data = data[
            (data['fnid'] == target_region) & 
            (data['harvest_year'] == target_year)
        ]
        
        if len(target_data) > 0:
            # Get actual yield
            actual_yield = target_data['yield'].values[0]
            
            # Predict counterfactual
            intervention = intervention_map[intervention_type]
            counterfactual_yield = model.predict_counterfactual(target_data, intervention)[0]
            
            # Calculate improvement
            improvement = counterfactual_yield - actual_yield
            improvement_pct = (improvement / actual_yield) * 100
            
            # Display results
            st.success("Analysis Complete!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Actual Yield",
                    value=f"{actual_yield:.2f} tons/ha",
                    delta=None
                )
            
            with col2:
                st.metric(
                    label="Counterfactual Yield",
                    value=f"{counterfactual_yield:.2f} tons/ha",
                    delta=f"+{improvement:.2f} tons/ha"
                )
            
            with col3:
                st.metric(
                    label="Improvement",
                    value=f"+{improvement_pct:.1f}%",
                    delta=None
                )
            
            # Visualization
            st.subheader("Impact Visualization")
            fig, ax = plt.subplots(figsize=(8, 6))
            
            scenarios = ['Actual', 'Counterfactual']
            yields = [actual_yield, counterfactual_yield]
            colors = ['#ff6b6b', '#51cf66']
            
            bars = ax.bar(scenarios, yields, color=colors, alpha=0.8)
            ax.set_ylabel('Yield (tons/ha)')
            ax.set_title(f'Counterfactual Analysis: {target_region} ({target_year})')
            
            # Add value labels on bars
            for bar, yield_val in zip(bars, yields):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                       f'{yield_val:.2f}', ha='center', va='bottom')
            
            st.pyplot(fig)
            
            # Interpretation
            st.subheader("💡 Interpretation")
            st.write(f"""
            If **{intervention_type.lower()}** had occurred in **{target_region}** during **{target_year}**:
            - Maize yields would have been **{improvement_pct:.1f}% higher**
            - That's an additional **{improvement:.2f} tons per hectare**
            - Total potential gain: **{improvement:.2f} tons/ha** × [area in hectares]
            """)
            
        else:
            st.error("No data found for the selected region and year.")

# Additional information
with st.expander("ℹ️ About This Tool"):
    st.markdown("""
    **How it works:**
    1. The model learns historical relationships between climate conditions, conflicts, and crop yields
    2. When you select an intervention, it simulates what would have happened under those new conditions
    3. The difference between actual and counterfactual yields shows the impact of the shock
    
    **Data Sources:**
    - Crop statistics: HarvestStat Africa
    - Climate data: Sample synthetic data (replace with CHIRPS, ERA5, etc.)
    - Conflict data: Sample synthetic data (replace with ACLED, etc.)
    
    **Methodology:**
    - Machine learning: Random Forest Regressor
    - Causal inference: Counterfactual prediction
    - Validation: Out-of-sample testing
    """)

with st.expander("🚀 Future Enhancements"):
    st.markdown("""
    - [ ] Add real climate data from CHIRPS/ERA5
    - [ ] Incorporate actual conflict data from ACLED
    - [ ] Expand to all African countries and crops
    - [ ] Add satellite imagery integration
    - [ ] Create regional vulnerability maps
    - [ ] Add economic impact calculations
    """)
