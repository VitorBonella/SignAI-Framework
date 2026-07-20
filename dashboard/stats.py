import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, t
import streamlit as st

def corrected_ttest(data_a, data_b, n_outer_folds):
    """
    Computes the Corrected Resampled Student's t-test.
    Reference: Nadeau and Bengio (2003).
    """
    differences = np.array(data_a) - np.array(data_b)
    n = len(differences)
    mean_diff = np.mean(differences)
    var_diff = np.var(differences, ddof=1)
    
    # Correction term: 1/n + (n_test / n_train)
    # For k-fold CV: n_test/n_train = 1/(k-1)
    # n is the number of samples (rounds)
    test_train_ratio = 1 / (n_outer_folds - 1)
    
    denominator = np.sqrt((1/n + test_train_ratio) * var_diff)
    
    if denominator == 0:
        return 0.0, 1.0 # No difference
        
    t_stat = mean_diff / denominator
    p_value = 2 * (1 - t.cdf(abs(t_stat), df=n-1))
    
    return t_stat, p_value

def perform_statistical_analysis(df_dataset, metric, alpha=0.05):
    """
    Performs pairwise comparisons between methods using Wilcoxon and Corrected T-test.
    """
    # Pivot to have unique observations (Round + Fold) as index and methods as columns
    pivot_df = df_dataset.pivot(index='Observation_ID', columns='Method', values=metric)
    methods = pivot_df.columns.tolist()
    
    results = []
    
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            m1, m2 = methods[i], methods[j]
            
            # Pair data and remove NaNs
            paired = pivot_df[[m1, m2]].dropna()
            total_n = len(paired)
            
            if total_n < 2:
                continue
                
            # Always Wilcoxon
            try:
                stat_w, p_w = wilcoxon(paired[m1], paired[m2])
            except ValueError: # All differences zero
                stat_w, p_w = 0, 1.0
                
            # Corrected T-test logic
            p_t = np.nan
            t_stat = np.nan
            t_error = None
            
            # Get config to determine N = n_inner * n_outer
            config1 = df_dataset[df_dataset['Method'] == m1]['Config'].iloc[0]
            n_outer = config1.get('n_outer_folds', 1)
            n_inner = config1.get('n_inner_folds', 1)
            
            # The user defined N results as n_inner * n_outer
            N_results = n_inner * n_outer
            
            # Apply threshold check based on user instruction (N_Obs >= 30)
            if total_n < 30:
                t_error = f"N ({total_n}) < 30 (total observations)"
            else:
                if n_outer <= 1:
                    t_error = "n_outer_folds must be > 1"
                else:
                    t_stat, p_t = corrected_ttest(paired[m1], paired[m2], n_outer)
            
            mean_diff = paired[m1].mean() - paired[m2].mean()
            
            results.append({
                'Method A': m1,
                'Method B': m2,
                'N_Obs': total_n,
                'Mean Diff': mean_diff,
                'Wilcoxon p': p_w,
                'Corrected T p': p_t,
                'T-test Status': t_error if t_error else "Applied",
                'Significant?': "Yes" if (p_w < alpha or (not np.isnan(p_t) and p_t < alpha)) else "No"
            })
            
    return pd.DataFrame(results)

def generate_latex_table(stats_df, summary_df):
    """Generates LaTeX code for the results."""
    # Summary Table LaTeX
    summary_latex = summary_df.to_latex(
        index=False, 
        float_format="%.4f",
        caption="Summary of performance metrics (Mean $\pm$ Std)",
        label="tab:summary_metrics",
        escape=False
    )
    
    # Stats Table LaTeX
    if not stats_df.empty:
        stats_latex = stats_df.to_latex(
            index=False, 
            float_format="%.4f",
            caption="Paired statistical comparison (Wilcoxon and Corrected T-test)",
            label="tab:statistical_analysis",
            escape=False
        )
    else:
        stats_latex = "% Not enough data for statistical analysis"
        
    return summary_latex, stats_latex
