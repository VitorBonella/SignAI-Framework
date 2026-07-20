import streamlit as st
import pandas as pd
import numpy as np
from dashboard.data import get_results_dirs, load_all_results, generate_mock_data
from dashboard.stats import perform_statistical_analysis, generate_latex_table
from dashboard.plots import plot_boxplots, plot_mean_std

def main():
    st.set_page_config(layout="wide", page_title="SignalAI Benchmarking Dashboard")
    
    st.title("🔬 SignalAI: Benchmarking Dashboard")
    st.markdown("""
    This dashboard provides an interactive interface for visualizing and analyzing experimental results 
    from vibration-signal fault diagnosis benchmarks.
    """)
    
    # --- SIDEBAR: CONFIGURATION ---
    st.sidebar.header("🛠️ Configuration")
    
    data_mode = st.sidebar.radio("Data Mode", ["Real Results", "Mock Data (Test)"])
    
    if data_mode == "Real Results":
        res_dirs = get_results_dirs()
        if not res_dirs:
            st.error("No results directories found. Using Mock Data instead.")
            df = generate_mock_data()
        else:
            with st.spinner("Loading real data..."):
                df = load_all_results(res_dirs)
    else:
        st.sidebar.info("Generating mock fold-level data for testing.")
        df = generate_mock_data()
        
    if df.empty:
        st.error("The dataset is empty. Please check your result files.")
        st.stop()
        
    # --- FILTERS ---
    st.sidebar.subheader("🔍 Filters")
    
    datasets = sorted(df['Dataset'].unique())
    selected_dataset = st.sidebar.selectbox("Select Dataset", datasets)
    
    df_filtered = df[df['Dataset'] == selected_dataset]
    
    available_methods = sorted(df_filtered['Method'].unique())
    selected_methods = st.sidebar.multiselect(
        "Select Methods to Compare", 
        available_methods, 
        default=available_methods[:3] if len(available_methods) > 3 else available_methods
    )
    
    metric = st.sidebar.selectbox("Select Metric", ["Accuracy", "F1_Score"])
    
    if not selected_methods:
        st.warning("Please select at least one method to visualize.")
        st.stop()
        
    df_viz = df_filtered[df_filtered['Method'].isin(selected_methods)]
    
    # --- MAIN PAGE: VISUALIZATIONS ---
    st.header(f"📊 Results for: {selected_dataset}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribution Analysis")
        fig_box = plot_boxplots(df_viz, metric)
        st.pyplot(fig_box, width='stretch')
        
    with col2:
        st.subheader("Performance Summary")
        fig_mean = plot_mean_std(df_viz, metric)
        st.pyplot(fig_mean, width='stretch')
        
    # --- SECTION: STATISTICAL ANALYSIS ---
    st.divider()
    st.header("⚖️ Statistical Comparison")
    
    with st.expander("About the tests"):
        st.markdown("""
        - **Wilcoxon Signed-Rank Test:** A non-parametric test for paired samples. 
        - **Corrected Resampled Student's t-test:** A parametric test that accounts for the overlap in training sets 
          common in cross-validation (Nadeau and Bengio, 2003). 
          *Note: Only applied if $N \ge 30$ and fold information is available.*
        """)
        
    stats_df = perform_statistical_analysis(df_viz, metric)
    
    if stats_df.empty:
        st.info("Not enough data or methods selected for paired statistical analysis.")
    else:
        # Style the significant rows for high readability
        def highlight_sig(row):
            # Using a soft green background with explicit dark text color for contrast
            if row['Significant?'] == "Yes":
                return ['background-color: #90ee90; color: #000000; font-weight: bold' for _ in row]
            return ['' for _ in row]
            
        st.dataframe(stats_df.style.apply(highlight_sig, axis=1), width='stretch')
        
    # --- SECTION: LATEX EXPORT ---
    st.divider()
    st.header("📄 Export to LaTeX")
    
    # Generate summary df for LaTeX
    summary_df = df_viz.groupby('Method')[metric].agg(['mean', 'std']).reset_index()
    summary_df.columns = ['Method', 'Mean', 'Std']
    
    summary_latex, stats_latex = generate_latex_table(stats_df, summary_df)
    
    tex_col1, tex_col2 = st.columns(2)
    
    with tex_col1:
        st.subheader("Summary Table (Mean ± Std)")
        st.text_area("Copy LaTeX Code", summary_latex, height=250)
        
    with tex_col2:
        st.subheader("Statistical Analysis Table")
        st.text_area("Copy LaTeX Code", stats_latex, height=250)

if __name__ == "__main__":
    main()
