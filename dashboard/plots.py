import numpy as np
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

def plot_classifier_feature_heatmap(
    df, metric, methods=None, features=None, title=None,
    classifier_labels=None, dataset_labels=None, feature_set_labels=None,
):
    """
    Generates a mean +/- std heatmap of `metric` (Dataset x Feature_Group),
    faceted by classifier (Method_Name), laid out 2 panels per row (last
    odd panel centered). The single best classifier/feature-set combination
    for each dataset (row), across ALL selected classifiers, is highlighted
    with a black cell.

    `methods` / `features` filter (and, for `features`, order) which classifiers
    and feature sets are included. `classifier_labels` / `dataset_labels` /
    `feature_set_labels` are optional {raw_name: display_name} dicts used only
    for on-figure text (titles/ticks) — data lookup still uses raw names.
    `title` overrides the auto-generated figure title.

    Returns (fig, caption_text).
    """
    set_publication_style()

    metric_label = "Macro F1" if metric == "F1_Score" else "Balanced Accuracy"
    classifier_labels = classifier_labels or {}
    dataset_labels = dataset_labels or {}
    feature_set_labels = feature_set_labels or {}

    available_methods = sorted(df["Method_Name"].dropna().unique())
    if methods is None:
        methods = available_methods
    else:
        methods = [m for m in methods if m in available_methods]
    if not methods:
        raise ValueError("No classifiers available for heatmap.")

    datasets = sorted(df["Dataset"].dropna().unique())

    present_features = df["Feature_Group"].dropna().unique().tolist()
    if features is None:
        canonical_feature_order = ["time", "frequency", "psd", "wavelet", "spectral_envelope", "all"]
        features = [f for f in canonical_feature_order if f in present_features]
        features += sorted(f for f in present_features if f not in canonical_feature_order)
    else:
        features = [f for f in features if f in present_features]
    if not features:
        raise ValueError("No feature sets available for heatmap.")

    n_datasets = len(datasets)
    n_features = len(features)

    grouped = df.groupby(["Method_Name", "Dataset", "Feature_Group"])[metric].agg(["mean", "std"])

    means, stds = {}, {}
    for m in methods:
        mean_arr = np.full((n_datasets, n_features), np.nan)
        std_arr = np.full((n_datasets, n_features), np.nan)
        for di, ds in enumerate(datasets):
            for fi, feat in enumerate(features):
                key = (m, ds, feat)
                if key in grouped.index:
                    mean_val = grouped.loc[key, "mean"] * 100
                    std_val = grouped.loc[key, "std"] * 100
                    if np.isnan(std_val):
                        std_val = 0.0
                    mean_arr[di, fi] = mean_val
                    std_arr[di, fi] = std_val
        means[m] = mean_arr
        stds[m] = std_arr

    stacked = np.stack([means[m] for m in methods], axis=0)  # (n_clf, n_datasets, n_features)
    best_per_row = []
    for r in range(n_datasets):
        row_vals = stacked[:, r, :]
        if np.all(np.isnan(row_vals)):
            best_per_row.append(None)
        else:
            best_per_row.append(np.unravel_index(np.nanargmax(row_vals), row_vals.shape))

    n = len(methods)
    ncols_layout = min(n, 2)
    nrows_layout = int(np.ceil(n / 2))

    mosaic = []
    idx = 0
    for _ in range(nrows_layout):
        remaining = n - idx
        if remaining >= 2:
            mosaic.append([methods[idx], methods[idx + 1]])
            idx += 2
        else:
            mosaic.append([methods[idx], "."])
            idx += 1

    panel_w = 0.9 * n_features + 3.2
    panel_h = 0.85 * n_datasets + 1.6
    fig, axd = plt.subplot_mosaic(
        mosaic,
        figsize=(panel_w * ncols_layout, panel_h * nrows_layout),
        gridspec_kw={"wspace": 0.35, "hspace": 0.55},
    )

    feature_labels = [feature_set_labels.get(f, f.replace("_", " ").title()) for f in features]
    dataset_tick_labels = [dataset_labels.get(d, d) for d in datasets]

    last_ax = None
    for m in methods:
        ax = axd[m]
        mean_arr, std_arr = means[m], stds[m]
        mask = np.isnan(mean_arr)

        annot = np.full(mean_arr.shape, "", dtype=object)
        non_masked_order = []
        for i in range(n_datasets):
            for j in range(n_features):
                if not mask[i, j]:
                    annot[i, j] = f"{mean_arr[i, j]:.2f}\n±{std_arr[i, j]:.2f}"
                    non_masked_order.append((i, j))
        text_index = {cell: k for k, cell in enumerate(non_masked_order)}

        sns.heatmap(
            mean_arr, ax=ax, annot=annot, fmt="", cmap="YlGnBu", vmin=0, vmax=100,
            xticklabels=feature_labels, yticklabels=dataset_tick_labels,
            mask=mask, cbar=False, linewidths=0.5, linecolor="white",
            annot_kws={"size": 10},
        )
        ax.set_facecolor("#d9d9d9")
        ax.set_title(classifier_labels.get(m, m), fontsize=15, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="y", rotation=0, labelsize=10)
        ax.tick_params(axis="x", labelsize=10, rotation=20)

        ci = methods.index(m)
        for r, best in enumerate(best_per_row):
            if best is None:
                continue
            best_ci, c = best
            if best_ci != ci:
                continue
            ax.add_patch(plt.Rectangle((c, r), 1, 1, facecolor="black", edgecolor="white", lw=1.5, zorder=1))
            ax.texts[text_index[(r, c)]].set_color("white")

        last_ax = ax

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(last_ax.collections[0], cax=cbar_ax, label=f"{metric_label} (%)")

    fig.suptitle(title or f"{metric_label} (%) — mean ± std by classifier and feature set", fontsize=15)
    fig.tight_layout(rect=[0, 0, 0.9, 0.95])

    # Center a dangling last-row single panel under the row above it
    if n % 2 == 1 and n > 1:
        first_row = mosaic[0]
        span_left = axd[first_row[0]].get_position().x0
        right_name = first_row[1] if first_row[1] != "." else first_row[0]
        span_right = axd[right_name].get_position().x1
        center = (span_left + span_right) / 2
        last_name = methods[-1]
        pos_last = axd[last_name].get_position()
        axd[last_name].set_position([center - pos_last.width / 2, pos_last.y0, pos_last.width, pos_last.height])

    display_methods = [classifier_labels.get(m, m) for m in methods]
    caption = (
        "Heatmap of {} (%) mean $\\pm$ std across datasets and feature sets, faceted by classifier "
        "({}). The black cell in each dataset row marks the single best classifier/feature-set "
        "combination for that dataset.".format(metric_label, ", ".join(display_methods))
    )

    return fig, caption


def plot_radar_chart(df, zoom=True):
    """
    Generates a publication-ready radar (polar) chart comparing methods on
    Macro F1 (blue) and Balanced Accuracy (red).

    Each axis of the radar is a Method; the two overlaid polygons show the
    mean Macro F1 and mean Balanced Accuracy for that method across all
    rounds/folds.

    Returns (fig, caption_text).
    """
    set_publication_style()

    metric_labels = {'F1_Score': 'Macro F1', 'Accuracy': 'Balanced Accuracy'}
    colors = {'F1_Score': '#1f4e96', 'Accuracy': '#c0392b'}  # blue, red

    summary = df.groupby('Method')[['F1_Score', 'Accuracy']].mean().sort_index()
    methods = summary.index.tolist()
    n = len(methods)

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    all_vals = summary[['F1_Score', 'Accuracy']].to_numpy().flatten()
    if zoom and len(all_vals) > 0:
        lo = max(0.0, np.floor(all_vals.min() * 20) / 20 - 0.05)
        hi = min(1.0, np.ceil(all_vals.max() * 20) / 20 + 0.02)
        if hi - lo < 1e-6:
            lo, hi = 0.0, 1.0
    else:
        lo, hi = 0.0, 1.0
    ax.set_ylim(lo, hi)

    yticks = np.linspace(lo, hi, 5)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{v:.2f}" for v in yticks], fontsize=9)
    rlabel_angle = np.degrees(angles[0] + (angles[1] - angles[0]) / 2) if n > 1 else 45
    ax.set_rlabel_position(rlabel_angle)

    for metric in ['F1_Score', 'Accuracy']:
        values = summary[metric].tolist()
        values += values[:1]
        ax.plot(angles, values, color=colors[metric], linewidth=2,
                label=metric_labels[metric], marker='o', markersize=5)
        ax.fill(angles, values, color=colors[metric], alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(methods)
    ax.tick_params(axis='x', pad=15)
    ax.set_title("Method Comparison: Macro F1 vs. Balanced Accuracy", pad=30, y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1))

    caption = (
        "Radar chart comparing {} on mean Macro F1 (blue) and mean Balanced Accuracy "
        "(red), averaged across all evaluation rounds/folds.".format(
            ", ".join(methods)
        )
    )

    plt.tight_layout()
    return fig, caption
