import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def set_publication_style():
    """Sets the plotting style to be publication-ready."""
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.dpi': 300,
        'axes.grid': True,
        'grid.alpha': 0.3
    })

def plot_boxplots(df, metric):
    """Generates a boxplot distribution for the selected metric."""
    set_publication_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort methods by mean performance for better visualization
    order = df.groupby('Method')[metric].mean().sort_values(ascending=False).index
    
    sns.boxplot(
        data=df, 
        x=metric, 
        y='Method', 
        hue='Method',
        legend=False,
        ax=ax, 
        order=order,
        palette="viridis",
        width=0.6,
        showmeans=True,
        meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"8"}
    )
    
    ax.set_title(f"Distribution of {metric} across Runs", pad=20)
    ax.set_xlabel(f"{metric} Value")
    ax.set_ylabel("Method / Configuration")
    
    plt.tight_layout()
    return fig

def plot_mean_std(df, metric):
    """Generates a bar plot with error bars for mean and std."""
    set_publication_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate stats
    stats = df.groupby('Method')[metric].agg(['mean', 'std']).reset_index()
    stats = stats.sort_values(by='mean', ascending=False)
    
    sns.barplot(
        data=stats, 
        x='mean', 
        y='Method', 
        hue='Method',
        legend=False,
        ax=ax, 
        palette="magma",
        capsize=.1,
        err_kws={'linewidth': 1.5}
    )
    
    # Add std error bars manually if barplot errorbars aren't enough or need custom styling
    # In Seaborn 0.13+, xerr is deprecated in barplot, it uses 'errorbar'
    
    ax.errorbar(
        x=stats['mean'], 
        y=stats['Method'], 
        xerr=stats['std'], 
        fmt='none', 
        c='black', 
        capsize=5
    )
    
    ax.set_title(f"Mean {metric} with Standard Deviation", pad=20)
    ax.set_xlabel(f"Mean {metric}")
    ax.set_ylabel("Method / Configuration")
    
    # Annotate bars with mean values
    for i, p in enumerate(ax.patches):
        width = p.get_width()
        ax.text(width + 0.01, p.get_y() + p.get_height()/2, f'{width:.3f}', 
                va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    return fig
