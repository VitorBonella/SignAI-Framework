import os
import json
import pandas as pd
import numpy as np
import streamlit as st

def get_results_dirs():
    """Returns a list of potential result directories."""
    # Priority order: user-specified path, project results folder, framework experiments results
    dirs = ["results", "signalAI/Experiments"]
    return [d for d in dirs if os.path.exists(d)]

@st.cache_data
def load_all_results(base_dirs):
    """
    Crawls directories to find JSON results.
    Expected structure: <base_dir>/<Dataset>/<Feature_Group>/<Method>/round<N>.json
    """
    data = []
    for base in base_dirs:
        for root, dirs, files in os.walk(base):
            for file in files:
                if file.endswith(".json") and (file.startswith("round") or file == "results.json"):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r') as f:
                            content = json.load(f)
                            
                            # Extract path-based metadata
                            # root: results/CWRU_12K/all/svm
                            parts = root.split(os.sep)
                            # dataset is usually the first folder after base
                            # parts for 'results/CWRU_12K/all/svm' with base='results'
                            # would be ['results', 'CWRU_12K', 'all', 'svm']
                            base_parts = base.split(os.sep)
                            relative_parts = parts[len(base_parts):]
                            
                            if len(relative_parts) < 3:
                                continue
                                
                            dataset = relative_parts[0]
                            feature_group = relative_parts[1]
                            method = relative_parts[2]
                            
                            metrics_overall = content.get('overall_metrics', {})
                            config = content.get('config', {})
                            folds = content.get('folds', [])
                            
                            if file.startswith("round"):
                                round_str = file.replace("round", "").replace(".json", "")
                                try:
                                    round_num = int(round_str)
                                except:
                                    round_num = round_str
                            else:
                                round_num = 1

                            # Extract each fold as a separate observation
                            for fold in folds:
                                fold_idx = fold.get('fold_index', 0)
                                fold_metrics = fold.get('metrics', {})
                                
                                data.append({
                                    'Dataset': dataset,
                                    'Method': f"{method} ({feature_group})",
                                    'Method_Name': method,
                                    'Feature_Group': feature_group,
                                    'Round': round_num,
                                    'Fold': fold_idx,
                                    'Observation_ID': f"R{round_num}_F{fold_idx}",
                                    'Accuracy': fold_metrics.get('accuracy', np.nan),
                                    'F1_Score': fold_metrics.get('f1', np.nan),
                                    'Config': config
                                })
                    except Exception as e:
                        # Silently skip malformed files during discovery
                        continue
    return pd.DataFrame(data)

def generate_mock_data(n_datasets=2, n_methods=3, n_rounds=8, n_outer=5, n_inner=7):
    """Generates synthetic benchmarking data for testing."""
    datasets = [f"Dataset_{i}" for i in range(1, n_datasets + 1)]
    methods = [("SVM", "all"), ("RandomForest", "time"), ("CNN", "psd")]
    
    data = []
    for ds in datasets:
        for method_name, feat in methods:
            # Baseline performance
            base_acc = np.random.uniform(0.7, 0.9)
            base_f1 = base_acc - np.random.uniform(0, 0.05)
            
            for r in range(1, n_rounds + 1):
                for f in range(n_outer):
                    acc = base_acc + np.random.normal(0, 0.05)
                    f1 = base_f1 + np.random.normal(0, 0.05)
                    
                    data.append({
                        'Dataset': ds,
                        'Method': f"{method_name} ({feat})",
                        'Method_Name': method_name,
                        'Feature_Group': feat,
                        'Round': r,
                        'Fold': f,
                        'Observation_ID': f"R{r}_F{f}",
                        'Accuracy': np.clip(acc, 0, 1),
                        'F1_Score': np.clip(f1, 0, 1),
                        'Config': {
                            'n_outer_folds': n_outer,
                            'n_inner_folds': n_inner
                        }
                    })
    return pd.DataFrame(data)
