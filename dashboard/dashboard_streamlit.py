"""
File: dashboard/dashboard_streamlit.py

Macro Transmission Monitor - Interactive Dashboard (v2.1)
Upgrade: interactive analytics, clustered heatmaps, and regime exploration.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dashboard_api import (
    get_returns_data,
    get_pairing_matrix,
    get_thresholds,
    get_latest_date
)

# =============================================================================
# 1. UI CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Macro Monitor | Institutional",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🌐"
)

# Dark Mode CSS & Custom Styling
st.markdown("""
    <style>
        .stApp {background-color: #0E1117;}
        h1, h2, h3 {font-family: 'Roboto Mono', monospace; color: #e0e0e0;}
        /* Removed fixed font size for metrics to prevent alignment issues */
        div[data-testid="stMetricValue"] {font-family: 'Roboto Mono', monospace;}
        div[data-testid="stVerticalBlock"] > div {
            border: 1px solid #333; background-color: #161b22; padding: 15px; border-radius: 4px;
        }
        /* Custom alert styling */
        .critical-alert { border-left: 5px solid #ff4b4b; padding-left: 10px; }
        .high-alert { border-left: 5px solid #ffa700; padding-left: 10px; }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. ANALYTICS ENGINE (Client-Side)
# =============================================================================

@st.cache_data(ttl=3600)
def load_data():
    """Load returns data efficiently."""
    returns_df = get_returns_data()
    pairings = get_pairing_matrix()
    return returns_df, pairings

def calculate_market_state(pairings, returns_df, z_threshold, short_window):
    """
    Dynamic analytics engine. Recomputes rolling correlations based on user sliders.
    """
    results = []
    long_window = 180

    if returns_df.empty:
        return pd.DataFrame()

    for pair in pairings:
        macro = pair["macro"]
        target = pair["target"]

        if macro not in returns_df.columns or target not in returns_df.columns:
            continue

        m = returns_df[macro]
        t = returns_df[target]

        # Rolling Correlation
        # Handle cases where window > data length
        if len(m) < short_window:
            continue
            
        rolling_corr = m.rolling(short_window).corr(t)
        
        # Get latest values
        if len(rolling_corr) < 1:
            continue
            
        curr_corr = rolling_corr.iloc[-1]
        
        # Baseline (last 180 days of the rolling correlation series)
        baseline = rolling_corr.iloc[-long_window:]
        mean_corr = baseline.mean()
        std_corr = baseline.std()
        
        if pd.isna(std_corr) or std_corr < 1e-4:
            z_score = 0.0
        else:
            z_score = (curr_corr - mean_corr) / std_corr

        # Status Logic
        if abs(z_score) >= z_threshold:
            status = "BREAK"
            severity = "CRITICAL"
        elif abs(z_score) >= max(1.5, z_threshold * 0.75):
            status = "STRESS"
            severity = "HIGH"
        else:
            status = "NOMINAL"
            severity = "LOW"
            
        results.append({
            "Macro": macro,
            "Target": target,
            "Status": status,
            "Severity": severity,
            "Z_Score": z_score,
            "Curr_Corr": curr_corr,
            "Rationale": pair.get("rationale", "N/A")
        })

    return pd.DataFrame(results)

# =============================================================================
# 3. CHARTING
# =============================================================================

def plot_regime_structure(returns_df, macro, target, window):
    """Plot correlation regime vs cumulative returns."""
    subset = returns_df.iloc[-252:]  # Last year
    
    if subset.empty:
        return go.Figure()

    corr = subset[macro].rolling(window).corr(subset[target])
    
    # Calculate Cumulative Returns (for Log Returns: exp(cumsum))
    # Fill NaN with 0 for cumsum to avoid breaking the line
    cum_macro = np.exp(subset[macro].fillna(0).cumsum())
    cum_target = np.exp(subset[target].fillna(0).cumsum())
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Correlation Area (RHS)
    fig.add_trace(go.Scatter(
        x=subset.index, y=corr, name=f"{window}D Corr",
        line=dict(color='rgba(150, 150, 150, 0.5)', width=1),
        fill='tozeroy', fillcolor='rgba(150, 150, 150, 0.1)'
    ), secondary_y=True)
    
    # Cumulative Returns (LHS)
    fig.add_trace(go.Scatter(
        x=subset.index, y=cum_macro, name=macro, 
        line=dict(color='#00CC96', width=2)
    ), secondary_y=False)
    
    fig.add_trace(go.Scatter(
        x=subset.index, y=cum_target, name=target, 
        line=dict(color='#AB63FA', width=2)
    ), secondary_y=False)
    
    fig.update_layout(
        template="plotly_dark", 
        height=400, 
        margin=dict(l=0, r=0, t=20, b=20),
        legend=dict(orientation="h", y=-0.15, x=0), 
        hovermode="x unified",
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        # title removed to prevent overlap with legend
    )
    
    fig.update_yaxes(title="Cumulative Return", showgrid=False, secondary_y=False)
    fig.update_yaxes(title="Correlation", showgrid=True, gridcolor='#333', range=[-1.1, 1.1], secondary_y=True)
    
    return fig

# =============================================================================
# 4. MAIN APP
# =============================================================================

def main():
    st.sidebar.markdown("## 🎛️ Analytics Config")
    st.sidebar.info("Adjust parameters to filter noise vs. signal.")
    
    # Config / Defaults
    default_thresholds = get_thresholds()
    
    # 1. Lookback Window
    window = st.sidebar.slider(
        "Correlation Window (Days)", 
        min_value=10, max_value=90, 
        value=int(default_thresholds.get("corr_window", 30)), 
        step=5,
        help="Short window = Noisy but fast. Long window = Stable but lagging."
    )
    
    # 2. Z-Score Slider
    z_thresh = st.sidebar.slider(
        "Breakdown Sensitivity (Z)", 
        min_value=1.5, max_value=4.0, 
        value=float(default_thresholds.get("z_threshold", 2.0)), 
        step=0.1,
        help="Higher Z = Only show extreme events."
    )
    
    st.sidebar.divider()
    if st.sidebar.button("Reload Data"):
        st.cache_data.clear()
        st.rerun()

    # --- Load Data ---
    returns_df, pairings = load_data()
    
    if returns_df.empty:
        st.error("No data available. Please run the backend pipeline first (`python src/macro_monitor.py --backfill`).")
        return

    # Run Analysis
    scan_df = calculate_market_state(pairings, returns_df, z_thresh, window)

    # --- Header Metrics ---
    def get_last_price(ticker):
        # We need raw prices for this, but currently only dealing with returns_df.
        # dashboard_api could fetch prices, but for speed let's just use returns context 
        # or skip this if too complex. The user's code had raw prices. 
        # We'll skip or use placeholders.
        return 0.0

    st.markdown("## 📊 Global Macro Transmission Monitor")
    
    # --- Tabs ---
    tab_risk, tab_matrix, tab_lab = st.tabs(["🚨 RISK RADAR", "🌐 TRANSMISSION MATRIX", "🔬 QUANT LAB"])

    # TAB 1: ALERTS
    with tab_risk:
        if scan_df.empty:
            st.info("No pairings configured.")
        else:
            breaks = scan_df[scan_df['Severity'] == 'CRITICAL']
            stress = scan_df[scan_df['Severity'] == 'HIGH']
            
            # KPI Cards
            c1, c2, c3 = st.columns(3)
            c1.metric("Critical Breaks", len(breaks), delta_color="inverse")
            c2.metric("Stressed Pairs", len(stress), delta_color="off")
            c3.metric("Scanned Pairs", len(scan_df))
            
            st.divider()

            if breaks.empty and stress.empty:
                st.success("System Nominal. All correlations within defined bounds.")
            else:
                # Show Critical First
                for _, row in breaks.iterrows():
                    with st.expander(f"🔴 CRITICAL: {row['Macro']} → {row['Target']} (Z={row['Z_Score']:.2f})", expanded=True):
                        st.markdown(f"**Rationale:** {row['Rationale']}")
                        st.plotly_chart(
                            plot_regime_structure(returns_df, row['Macro'], row['Target'], window), 
                            width="stretch",
                            key=f"chart_{row['Macro']}_{row['Target']}_{row['Severity']}"
                        )

                for _, row in stress.iterrows():
                    with st.expander(f"🟡 STRESS: {row['Macro']} → {row['Target']} (Z={row['Z_Score']:.2f})", expanded=False):
                        st.markdown(f"**Rationale:** {row['Rationale']}")
                        st.plotly_chart(
                            plot_regime_structure(returns_df, row['Macro'], row['Target'], window), 
                            width="stretch",
                            key=f"chart_{row['Macro']}_{row['Target']}_{row['Severity']}"
                        )

    # TAB 2: HEATMAP (CLUSTERED)
    with tab_matrix:
        st.markdown("### Structural Correlation Health")
        
        if not scan_df.empty:
            # 1. PIVOT
            pivot_z = scan_df.pivot(index='Macro', columns='Target', values='Z_Score')
            
            # 2. CLUSTERING ORDER (Rates -> FX -> Commodities -> Equities)
            # Adapt to our universe
            desired_order_macros = [
                'US_10Y', 'US_2Y', 'VIX',              # Rates/Vol
                'DXY', 'EURUSD', 'USDJPY',             # FX
                'Brent', 'WTI', 'Gold', 'Copper',      # Commodities
                'SP500', 'Nasdaq', 'DAX', 'Nifty50'    # Equities (Drivers)
            ]
            
            sorted_index = [x for x in desired_order_macros if x in pivot_z.index]
            remaining = [x for x in pivot_z.index if x not in sorted_index]
            final_index = sorted_index + remaining
            
            pivot_z = pivot_z.reindex(final_index)
            
            # 3. PLOT
            fig_hm = go.Figure(data=go.Heatmap(
                z=pivot_z.values,
                x=pivot_z.columns,
                y=pivot_z.index,
                colorscale='RdBu_r', zmin=-3, zmax=3,
                xgap=1, ygap=1,
                hovertemplate="Macro: %{y}<br>Target: %{x}<br>Z-Score: %{z:.2f}<extra></extra>"
            ))
            fig_hm.update_layout(
                template="plotly_dark", 
                height=700, 
                title="Clustered Transmission Matrix (Z-Score)",
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_hm, width="stretch")
        else:
            st.warning("No data for heatmap.")

    # TAB 3: QUANT LAB
    with tab_lab:
        st.markdown("### Interactive Regime Explorer")
        
        c1, c2 = st.columns(2)
        assets = sorted(returns_df.columns)
        
        if assets:
            sel_macro = c1.selectbox("Macro Driver", assets, index=0)
            # Try to pick a logical target default
            default_target_idx = 1 if len(assets) > 1 else 0
            sel_target = c2.selectbox("Target Asset", assets, index=default_target_idx)
            
            st.plotly_chart(
                plot_regime_structure(returns_df, sel_macro, sel_target, window), 
                width="stretch"
            )
            
            # Stats
            r_corr = returns_df[sel_macro].rolling(window).corr(returns_df[sel_target]).iloc[-1]
            st.metric(f"Current {window}-Day Correlation", f"{r_corr:.3f}")
        else:
            st.warning("No assets available.")

if __name__ == "__main__":
    main()
