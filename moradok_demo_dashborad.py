import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# --- Set page config ONCE at the very beginning ---
st.set_page_config(page_title="Demo Moradok Dashboard", layout="wide")

# --- Updated Title ---
st.title("📊 Demo Moradok Dashboard")
st.markdown("*Asset Portfolio Analysis and Visualization*")

# --- Sidebar Inputs ---
st.sidebar.title("Risk & Portfolio Options")
risk_model = st.sidebar.radio(
    "Risk Model:",
    ("Historical Volatility", "VaR", "Beta Model")
)

catalog_option = st.sidebar.selectbox(
    "Asset Catalog:",
    ("All Assets", "Technology", "Healthcare", "Finance")
)

st.sidebar.markdown("#### Custom Risk Calculation")
initial_price = st.sidebar.number_input("Initial Price ($):", min_value=1.0, value=100.0)
volatility = st.sidebar.slider("Expected daily volatility (%)", 0.1, 10.0, 2.0)

# --- Create Mock Asset Data ---
num_assets = 30
np.random.seed(42)  # For reproducibility
asset_names = [f"Asset {i}" for i in range(1, num_assets + 1)]
risks = np.random.randint(0, 101, size=num_assets)  # Random risk percentage between 0 and 100
returns = np.random.normal(0.05, 0.03, size=num_assets)  # Random expected returns
descriptions = [
    f"This is a description for {name} in the {catalog_option} catalog." 
    for name in asset_names
]
# Random latitudes/longitudes within a US-like area (for map)
lats = np.random.uniform(37, 42, size=num_assets)
lons = np.random.uniform(-80, -70, size=num_assets)

assets_df = pd.DataFrame({
    "Asset": asset_names,
    "Risk": risks,
    "Return": returns,
    "Description": descriptions,
    "Lat": lats,
    "Lon": lons
})

# --- Risk Filter ---
risk_threshold = st.slider("Select maximum risk % you're willing to handle:", 0, 100, 50)
filtered_assets = assets_df[assets_df["Risk"] <= risk_threshold]

st.markdown(f"#### Showing {len(filtered_assets)} asset(s) with Risk ≤ {risk_threshold}%")

# --- Function to Generate a More Volatile Performance Series ---
def generate_performance_series(days=20, init=100, daily_vol=2.0):
    # Daily volatility is used as the standard deviation percent
    returns = np.random.normal(loc=0, scale=daily_vol/100, size=days)
    price_series = init * (1 + returns).cumprod()
    return price_series

# --- Display Asset Info with Toggle ---
st.markdown("### Asset Details & Performance")

# Show/hide asset details toggle
show_asset_details = st.checkbox("Show detailed asset charts", value=False)

if show_asset_details:
    asset_performances = {}  # to store each asset's performance series for portfolio aggregation
    
    # For each asset, create a row with two columns
    for _, row in filtered_assets.iterrows():
        # Create two columns for each asset row
        col_info, col_chart = st.columns([1, 1])
        
        with col_info:
            st.write(f"**{row['Asset']}**")
            st.write(f"Risk: **{row['Risk']}%**")
            st.write(f"Expected Return: **{row['Return']:.2%}**")
            st.write(row['Description'])
            
        with col_chart:
            series = generate_performance_series(days=20, init=initial_price, daily_vol=volatility)
            performance_df = pd.DataFrame({
                "Day": range(1, 21),
                "Price": series
            })
            st.line_chart(performance_df.set_index("Day"))
            
        # Save the series for portfolio simulation
        asset_performances[row['Asset']] = series
else:
    # Even when not showing details, we need to generate performances for portfolio simulation
    asset_performances = {}
    for _, row in filtered_assets.iterrows():
        series = generate_performance_series(days=20, init=initial_price, daily_vol=volatility)
        asset_performances[row['Asset']] = series
    
    # Show a summary table instead
    st.dataframe(
        filtered_assets[["Asset", "Risk", "Return"]].set_index("Asset").style.format({
            "Return": "{:.2%}"
        }),
        use_container_width=True
    )

# --- Portfolio Performance & Map Section, side by side ---
st.markdown("### Portfolio Analysis")

# Create two columns for chart and map
chart_col, map_col = st.columns(2)

with chart_col:
    st.markdown("#### Portfolio Performance")
    if asset_performances:
        # Compute portfolio performance as an equally weighted average of asset performances
        performance_matrix = np.array(list(asset_performances.values()))
        # Average across assets (axis=0 gives the daily mean)
        portfolio_series = performance_matrix.mean(axis=0)
        
        portfolio_df = pd.DataFrame({
            "Day": range(1, 21),
            "Portfolio Value": portfolio_series
        })
        
        st.line_chart(portfolio_df.set_index("Day"))
        st.write("The portfolio performance is computed as an equally weighted average of the included asset charts.")
    else:
        st.write("No assets available for portfolio simulation with the current risk threshold.")
    
    # --- Efficient Frontier Plot ---
    st.markdown("#### Risk-Return Analysis with Efficient Frontier")
    
    # Generate efficient frontier data
    def generate_efficient_frontier():
        # Generate random risk and return points
        risks = np.linspace(5, 25, 100)
        # Simulate an efficient frontier curve (higher risk should generally yield higher returns)
        returns = 0.03 + 0.25 * np.sqrt(risks/100) + np.random.normal(0, 0.005, len(risks))
        return risks, returns
    
    # Plot scatter of assets and efficient frontier
    def plot_efficient_frontier():
        fig = go.Figure()
        
        # Plot assets
        fig.add_trace(go.Scatter(
            x=filtered_assets['Risk'],
            y=filtered_assets['Return'],
            mode='markers',
            name='Assets',
            marker=dict(
                size=10,
                color='blue',
            )
        ))
        
        # Generate and plot efficient frontier
        ef_risks, ef_returns = generate_efficient_frontier()
        fig.add_trace(go.Scatter(
            x=ef_risks,
            y=ef_returns,
            mode='lines',
            name='Efficient Frontier',
            line=dict(color='red', width=2)
        ))
        
        # Add a star for the optimal portfolio
        opt_risk = 15
        opt_return = 0.03 + 0.25 * np.sqrt(opt_risk/100)
        fig.add_trace(go.Scatter(
            x=[opt_risk],
            y=[opt_return],
            mode='markers',
            name='Optimal Portfolio',
            marker=dict(
                size=15,
                symbol='star',
                color='gold',
                line=dict(width=2, color='black')
            )
        ))
        
        fig.update_layout(
            title='Efficient Frontier Analysis',
            xaxis_title='Risk (%)',
            yaxis_title='Expected Return',
            yaxis_tickformat='.1%',
            height=500,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
    
    # Display the efficient frontier plot
    st.plotly_chart(plot_efficient_frontier(), use_container_width=True)

with map_col:
    st.markdown("#### Asset Locations Map")
    map_data = filtered_assets[["Lat", "Lon"]].rename(columns={"Lat": "lat", "Lon": "lon"})
    st.map(map_data)
    
    # Add a table of assets below the map
    st.markdown("#### Assets Summary")
    summary_df = filtered_assets[["Asset", "Risk", "Return"]].copy()
    summary_df["Return"] = summary_df["Return"].apply(lambda x: f"{x:.2%}")  # Format as percentage
    st.dataframe(summary_df, use_container_width=True)

# --- Bottom Menu Info ---
st.markdown(f"**Selected Risk Model:** {risk_model} | **Asset Catalog:** {catalog_option}")
st.markdown("---")
st.markdown("**Moradok Dashboard Demo** | © 2025")