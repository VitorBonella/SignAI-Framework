import io
import streamlit as st
import pandas as pd
import numpy as np
from dashboard.data import get_results_dirs, load_all_results, generate_mock_data
from dashboard.stats import perform_statistical_analysis, generate_latex_table, generate_latex_figure
from dashboard.plots import plot_boxplots, plot_mean_std, plot_radar_chart, plot_classifier_feature_heatmap

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

    # --- SECTION: RADAR CHART (PAPER EXPORT) ---
    st.divider()
    st.header("📡 Radar Chart: Macro F1 vs. Balanced Accuracy")

    zoom_radar = st.checkbox(
        "Zoom radial axis to data range (recommended when methods are close)",
        value=True
    )

    fig_radar, radar_caption = plot_radar_chart(df_viz, zoom=zoom_radar)
    st.pyplot(fig_radar, width='stretch')
    st.caption(radar_caption)

    png_buf = io.BytesIO()
    fig_radar.savefig(png_buf, format="png", dpi=300, bbox_inches="tight")
    pdf_buf = io.BytesIO()
    fig_radar.savefig(pdf_buf, format="pdf", bbox_inches="tight")

    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            "⬇️ Download PNG (300 DPI)",
            data=png_buf.getvalue(),
            file_name="radar_comparison.png",
            mime="image/png"
        )
    with dl_col2:
        st.download_button(
            "⬇️ Download PDF (vector, for LaTeX)",
            data=pdf_buf.getvalue(),
            file_name="radar_comparison.pdf",
            mime="application/pdf"
        )

    with st.expander("📄 LaTeX figure snippet"):
        radar_latex = generate_latex_figure(
            caption=radar_caption,
            label="fig:radar_comparison",
            filename="radar_comparison.pdf"
        )
        st.text_area("Copy LaTeX Code", radar_latex, height=180)

    # --- SECTION: CLASSIFIER x FEATURE-SET HEATMAP ---
    st.divider()
    st.header("🧩 Classifier × Feature-Set Heatmap")
    st.markdown(
        "Compact alternative to one results table per classifier: mean ± std per dataset and "
        "feature set, faceted by classifier. Uses **all datasets** (not just the one selected above)."
    )

    all_classifiers = sorted(df["Method_Name"].dropna().unique())
    canonical_feature_order = ["time", "frequency", "psd", "wavelet", "spectral_envelope", "all"]
    present_features = df["Feature_Group"].dropna().unique().tolist()
    all_features = [f for f in canonical_feature_order if f in present_features]
    all_features += sorted(f for f in present_features if f not in canonical_feature_order)
    all_datasets = sorted(df["Dataset"].dropna().unique())

    hm_sel_col1, hm_sel_col2 = st.columns(2)
    with hm_sel_col1:
        heatmap_classifiers = st.multiselect(
            "Classifiers to include", all_classifiers, default=all_classifiers
        )
    with hm_sel_col2:
        heatmap_features = st.multiselect(
            "Feature sets to include", all_features, default=all_features
        )

    with st.expander("✏️ Customize labels"):
        heatmap_title = st.text_input(
            "Figure title", value="", placeholder="(auto-generated if left blank)"
        )

        st.markdown("**Classifiers**")
        classifier_label_map = {}
        clf_cols = st.columns(min(len(heatmap_classifiers), 4)) if heatmap_classifiers else []
        for i, m in enumerate(heatmap_classifiers):
            with clf_cols[i % len(clf_cols)]:
                classifier_label_map[m] = st.text_input(m, value=m, key=f"clf_label_{m}")

        st.markdown("**Datasets**")
        dataset_label_map = {}
        ds_cols = st.columns(min(len(all_datasets), 4)) if all_datasets else []
        for i, d in enumerate(all_datasets):
            with ds_cols[i % len(ds_cols)]:
                dataset_label_map[d] = st.text_input(d, value=d.replace("_", " "), key=f"ds_label_{d}")

        st.markdown("**Feature sets**")
        feature_label_map = {}
        ft_cols = st.columns(min(len(heatmap_features), 4)) if heatmap_features else []
        for i, f in enumerate(heatmap_features):
            with ft_cols[i % len(ft_cols)]:
                feature_label_map[f] = st.text_input(
                    f, value=f.replace("_", " ").title(), key=f"ft_label_{f}"
                )

    if not heatmap_classifiers:
        st.warning("Select at least one classifier for the heatmap.")
    elif not heatmap_features:
        st.warning("Select at least one feature set for the heatmap.")
    else:
        fig_heatmap, heatmap_caption = plot_classifier_feature_heatmap(
            df, metric, heatmap_classifiers, heatmap_features,
            title=heatmap_title or None,
            classifier_labels=classifier_label_map,
            dataset_labels=dataset_label_map,
            feature_set_labels=feature_label_map,
        )
        st.pyplot(fig_heatmap, width='stretch')
        st.caption(heatmap_caption)

        png_dpi = st.select_slider(
            "PNG resolution (DPI)", options=[150, 300, 600, 900, 1200], value=1200,
            help="Vector formats (PDF/SVG) are already lossless at any zoom; this only affects the PNG."
        )
        heatmap_png_buf = io.BytesIO()
        fig_heatmap.savefig(heatmap_png_buf, format="png", dpi=png_dpi, bbox_inches="tight")
        heatmap_pdf_buf = io.BytesIO()
        fig_heatmap.savefig(heatmap_pdf_buf, format="pdf", bbox_inches="tight")
        heatmap_svg_buf = io.BytesIO()
        fig_heatmap.savefig(heatmap_svg_buf, format="svg", bbox_inches="tight")

        hm_col1, hm_col2, hm_col3 = st.columns(3)
        with hm_col1:
            st.download_button(
                f"⬇️ Download PNG ({png_dpi} DPI)",
                data=heatmap_png_buf.getvalue(),
                file_name="classifier_feature_heatmap.png",
                mime="image/png",
                key="heatmap_png"
            )
        with hm_col2:
            st.download_button(
                "⬇️ Download PDF (vector, for LaTeX)",
                data=heatmap_pdf_buf.getvalue(),
                file_name="classifier_feature_heatmap.pdf",
                mime="application/pdf",
                key="heatmap_pdf"
            )
        with hm_col3:
            st.download_button(
                "⬇️ Download SVG (vector, editable)",
                data=heatmap_svg_buf.getvalue(),
                file_name="classifier_feature_heatmap.svg",
                mime="image/svg+xml",
                key="heatmap_svg"
            )

        with st.expander("📄 LaTeX figure snippet"):
            heatmap_latex = generate_latex_figure(
                caption=heatmap_caption,
                label="fig:classifier_feature_heatmap",
                filename="classifier_feature_heatmap.pdf"
            )
            st.text_area("Copy LaTeX Code", heatmap_latex, height=180, key="heatmap_latex")

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
