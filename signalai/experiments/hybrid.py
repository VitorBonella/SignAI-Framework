import copy
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from signalai.models.autoencoder import DCAE1D
from signalai.utils.logging import setup_logger
from signalai.utils.metrics import calculate_metrics
from signalai.utils.results import ExperimentResults, FoldResults


CLASSIFIERS: Dict[str, Tuple] = {
    "rf": (
        RandomForestClassifier(random_state=42),
        {
            "n_estimators": [50, 100, 200],
            "criterion": ["gini", "entropy", "log_loss"],
            "max_depth": [10, 25, 50],
            "min_samples_split": [2, 5, 10],
        },
    ),
    "svm": (
        SVC(random_state=42, probability=True),
        {
            "C": [0.1, 1, 10, 100],
            "kernel": ["linear", "rbf", "poly"],
            "gamma": ["scale", "auto"],
        },
    ),
}


def extract_fold_features(checkpoint_path, X_split, device, batch_size=256):
    """Load DCAE checkpoint and return flattened encoder features for X_split."""
    model = DCAE1D()
    state_dict = torch.load(checkpoint_path, map_location=device)
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device).eval()
    parts = []
    with torch.no_grad():
        for i in range(0, len(X_split), batch_size):
            batch = torch.tensor(X_split[i : i + batch_size], dtype=torch.float32).to(device)
            parts.append(model.extract_features(batch).cpu().numpy())
    return np.concatenate(parts, axis=0)


def run_classifier_fold(X_train_feat, y_train, X_test_feat, y_test, clf, search_space, fold_val):
    scaler = StandardScaler()
    X_train_feat = scaler.fit_transform(X_train_feat)
    X_test_feat = scaler.transform(X_test_feat)

    inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    grid = GridSearchCV(
        copy.deepcopy(clf), search_space, cv=inner_cv, scoring="f1_macro", n_jobs=-1
    )
    grid.fit(X_train_feat, y_train)
    best = grid.best_estimator_

    y_pred = best.predict(X_test_feat)
    y_proba = best.predict_proba(X_test_feat) if hasattr(best, "predict_proba") else None
    metrics = calculate_metrics(y_test, y_pred, y_proba)

    return FoldResults(
        fold_index=fold_val,
        y_true=y_test,
        y_pred=y_pred,
        y_proba=np.array(y_proba) if y_proba is not None else None,
        metrics=metrics,
    )


def run_one_round(X, y, assignments, dcae_run_dir, clf_name, clf, search_space, device, round_label=None):
    """Run all outer folds for one round and return ExperimentResults."""
    dcae_run_dir = Path(dcae_run_dir)
    unique_folds = np.unique(assignments)
    prefix = f"round{round_label}_" if round_label is not None else ""

    results = ExperimentResults(
        experiment_name=f"dcae_{clf_name}",
        description=f"DCAE encoder features + {clf_name.upper()} classifier",
        model_name=clf.__class__.__name__,
        feature_names=None,
        config={
            "n_outer_folds": int(len(unique_folds)),
            "n_inner_folds": 4,
            "random_state": 42,
            "round": round_label,
        },
    )

    for fold_val in unique_folds:
        print(f"    [{clf_name.upper()}] Fold {fold_val}/{len(unique_folds) - 1}...")
        train_mask = assignments != fold_val
        test_mask = assignments == fold_val

        checkpoint_path = dcae_run_dir / f"{prefix}model_fold_{fold_val}.pth"
        X_train_feat = extract_fold_features(checkpoint_path, X[train_mask], device)
        X_test_feat = extract_fold_features(checkpoint_path, X[test_mask], device)

        fold_result = run_classifier_fold(
            X_train_feat, y[train_mask], X_test_feat, y[test_mask], clf, search_space, fold_val
        )
        results.add_fold_result(fold_result)

    results.calculate_overall_metrics()
    return results


def run_hybrid_classification(X, y, folds, dcae_run_dir, output_base, dataset, transform, device):
    """Run RF and SVM on DCAE features for all rounds and save results."""
    for clf_name, (clf, search_space) in CLASSIFIERS.items():
        print(f"\n=== DCAE + {clf_name.upper()} ===")
        clf_run_dir = (
            Path(output_base)
            / dataset
            / transform
            / f"dcae_{clf_name}"
        )
        clf_run_dir.mkdir(parents=True, exist_ok=True)
        logger = setup_logger(str(clf_run_dir / "experiment.log"))

        if isinstance(folds, list):
            for round_idx, round_folds in enumerate(folds):
                round_num = round_idx + 1
                print(f"\n  Round {round_num}/{len(folds)}")
                results = run_one_round(
                    X, y, round_folds, dcae_run_dir, clf_name, clf, search_space, device,
                    round_label=round_num,
                )
                results.save_json(clf_run_dir / f"round{round_num}.json")
        else:
            results = run_one_round(
                X, y, folds, dcae_run_dir, clf_name, clf, search_space, device,
            )
            results.save_json(clf_run_dir / "results.json")

        print(f"  Saved to {clf_run_dir}")
