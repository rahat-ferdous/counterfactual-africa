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
def ssp_analysis_tab():
    st.header("🌍 SSP Climate Scenario Analysis")
    st.markdown("""
    Analyze future agricultural outcomes under different Shared Socioeconomic Pathways (SSPs).
    These scenarios represent alternative futures with different climate and development trajectories.
    """)
    
    # Region selection
    regions = data['fnid'].unique()
    selected_region = st.selectbox("Select Region", regions)
    
    # Year selection
    target_year = st.slider("Target Year", 2025, 2050, 2030, 5)
    
    # SSP scenario selection
    selected_ssps = st.multiselect(
        "Select SSP Scenarios to Compare",
        ["SSP1 - Sustainability", "SSP2 - Middle Road", "SSP3 - Regional Rivalry", 
         "SSP4 - Inequality", "SSP5 - Fossil-Fueled"],
        default=["SSP1 - Sustainability", "SSP2 - Middle Road", "SSP3 - Regional Rivalry"]
    )
    
    # Map display names to internal names
    ssp_mapping = {
        "SSP1 - Sustainability": "SSP1",
        "SSP2 - Middle Road": "SSP2", 
        "SSP3 - Regional Rivalry": "SSP3",
        "SSP4 - Inequality": "SSP4",
        "SSP5 - Fossil-Fueled": "SSP5"
    }
    
    if st.button("Run SSP Analysis", type="primary"):
        with st.spinner("Running SSP scenario analysis..."):
            # Initialize SSP analyzer
            ssp_analyzer = SSPAnalyzer(model)
            ssp_analyzer.load_ssp_scenarios()
            
            # Run analysis
            region_data = data[data['fnid'] == selected_region]
            scenarios = ssp_analyzer.generate_future_scenarios(
                region_data, selected_region, [target_year])
            
            impacts = ssp_analyzer.calculate_ssp_impacts(selected_region)
            
            # Display results
            display_ssp_results(impacts, selected_ssps, ssp_mapping, target_year)

def display_ssp_results(impacts, selected_ssps, ssp_mapping, target_year):
    """Display SSP analysis results"""
    
    st.subheader("📊 Scenario Comparison")
    
    # Create comparison table
    comparison_data = []
    for display_name in selected_ssps:
        ssp = ssp_mapping[display_name]
        if ssp in impacts and target_year in impacts[ssp]:
            metrics = impacts[ssp][target_year]
            comparison_data.append({
                'Scenario': display_name,
                'Projected Yield': f"{metrics['mean_yield']:.2f} t/ha",
                'Change from Baseline': f"{metrics['yield_change_pct']:+.1f}%",
                'Drought Risk': f"{metrics['drought_risk']:.1%}",
                'Conflict Risk': f"{metrics['conflict_risk']:.1%}",
                'Severe Decline Probability': f"{metrics['probability_40pct_decline']:.1%}"
            })
    
    if comparison_data:
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
        
        # Visualization
        st.subheader("📈 Yield Projections by Scenario")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Yield projections
        scenarios = []
        yields = []
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
        
        for i, display_name in enumerate(selected_ssps):
            ssp = ssp_mapping[display_name]
            if ssp in impacts and target_year in impacts[ssp]:
                scenarios.append(display_name.split(' - ')[0])
                yields.append(impacts[ssp][target_year]['mean_yield'])
        
        bars = ax1.bar(scenarios, yields, color=colors[:len(scenarios)], alpha=0.8)
        ax1.set_ylabel('Yield (tons/ha)')
        ax1.set_title(f'Projected Maize Yields in {target_year}')
        ax1.tick_params(axis='x', rotation=45)
        
        for bar, yield_val in zip(bars, yields):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{yield_val:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Risk comparison
        risk_types = ['Drought', 'Conflict', 'Heat Stress']
        risk_data = {ssp: [] for ssp in scenarios}
        
        for display_name in selected_ssps:
            ssp = ssp_mapping[display_name]
            short_name = display_name.split(' - ')[0]
            if ssp in impacts and target_year in impacts[ssp]:
                risk_data[short_name] = [
                    impacts[ssp][target_year]['drought_risk'],
                    impacts[ssp][target_year]['conflict_risk'], 
                    impacts[ssp][target_year]['heat_risk']
                ]
        
        x = np.arange(len(risk_types))
        width = 0.8 / len(scenarios)
        
        for i, (scenario, risks) in enumerate(risk_data.items()):
            offset = width * i - width * (len(scenarios) - 1) / 2
            ax2.bar(x + offset, risks, width, label=scenario, 
                   color=colors[i], alpha=0.8)
        
        ax2.set_xlabel('Risk Type')
        ax2.set_ylabel('Probability')
        ax2.set_title('Risk Profile by Scenario')
        ax2.set_xticks(x)
        ax2.set_xticklabels(risk_types)
        ax2.legend()
        ax2.set_ylim(0, 1)
        
        # Plot 3: Yield change over time
        years = [2025, 2030, 2035, 2040, 2045, 2050]
        for display_name in selected_ssps:
            ssp = ssp_mapping[display_name]
            short_name = display_name.split(' - ')[0]
            changes = []
            for year in years:
                if ssp in impacts and year in impacts[ssp]:
                    changes.append(impacts[ssp][year]['yield_change_pct'])
            if changes:
                ax3.plot(years, changes, marker='o', label=short_name, linewidth=2)
        
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax3.set_xlabel('Year')
        ax3.set_ylabel('Yield Change (%)')
        ax3.set_title('Yield Trajectory 2025-2050')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Decline probabilities
        decline_data = []
        labels = []
        for display_name in selected_ssps:
            ssp = ssp_mapping[display_name]
            short_name = display_name.split(' - ')[0]
            if ssp in impacts and target_year in impacts[ssp]:
                decline_data.append([
                    impacts[ssp][target_year]['probability_20pct_decline'],
                    impacts[ssp][target_year]['probability_40pct_decline']
                ])
                labels.append(short_name)
        
        if decline_data:
            decline_data = np.array(decline_data)
            x = np.arange(len(labels))
            ax4.bar(x - 0.2, decline_data[:, 0], 0.4, label='>20% Decline', alpha=0.8)
            ax4.bar(x + 0.2, decline_data[:, 1], 0.4, label='>40% Decline', alpha=0.8)
            ax4.set_xlabel('Scenario')
            ax4.set_ylabel('Probability')
            ax4.set_title('Probability of Yield Declines')
            ax4.set_xticks(x)
            ax4.set_xticklabels(labels)
            ax4.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Policy recommendations
        st.subheader("💡 Policy Recommendations")
        
        best_scenario = max(selected_ssps, 
                          key=lambda x: impacts[ssp_mapping[x]][target_year]['mean_yield'])
        worst_scenario = min(selected_ssps, 
                           key=lambda x: impacts[ssp_mapping[x]][target_year]['mean_yield'])
        
        best_ssp = ssp_mapping[best_scenario]
        worst_ssp = ssp_mapping[worst_scenario]
        
        st.info(f"""
        **Key Insights for {selected_region}:**
        
        • **Most favorable scenario:** {best_scenario} ({impacts[best_ssp][target_year]['mean_yield']:.2f} t/ha)
        • **Most challenging scenario:** {worst_scenario} ({impacts[worst_ssp][target_year]['mean_yield']:.2f} t/ha)
        • **Performance gap:** {impacts[best_ssp][target_year]['mean_yield'] - impacts[worst_ssp][target_year]['mean_yield']:.2f} t/ha
        
        **Recommended focus areas:**
        {get_policy_recommendations(impacts, best_ssp, worst_ssp, target_year)}
        """)

def get_policy_recommendations(impacts, best_ssp, worst_ssp, target_year):
    """Generate policy recommendations based on scenario analysis"""
    
    recommendations = []
    
    # Analyze what makes the best scenario successful
    best_metrics = impacts[best_ssp][target_year]
    worst_metrics = impacts[worst_ssp][target_year]
    
    if best_metrics['drought_risk'] < worst_metrics['drought_risk']:
        recommendations.append("• **Invest in drought resilience** (irrigation, drought-tolerant crops)")
    
    if best_metrics['conflict_risk'] < worst_metrics['conflict_risk']:
        recommendations.append("• **Strengthen conflict prevention** and peacebuilding programs")
    
    if best_metrics['heat_risk'] < worst_metrics['heat_risk']:
        recommendations.append("• **Develop heat-tolerant crop varieties** and adjust planting calendars")
    
    if len(recommendations) == 0:
        recommendations.append("• **Focus on general agricultural development** and technology adoption")
    
    return "\n".join(recommendations)

# Add the new tab to your main app
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Counterfactual Analysis", "SSP Scenario Analysis"])
    
    if page == "Counterfactual Analysis":
        counterfactual_analysis_tab()
    else:
        ssp_analysis_tab()
