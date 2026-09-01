import sys
import argparse
import numpy as np
import os
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SequentialFeatureSelector, SelectKBest, f_classif

from lightgbm import LGBMClassifier

from signalai.features.selection import HybridRankingWrapperSelector, MRMRSelector
import vibdata.raw as raw_datasets
from vibdata.deep.DeepDataset import convertDataset
from vibdata.deep.signal.transforms import FilterByValue

from signalai.data.grouping import get_dataset_grouping
from signalai.sampling.generators import FoldIdxGeneratorUnbiased
from signalai.features.pipelines import get_pipeline, get_feature_names
from signalai.experiments.classification import ClassificationExperiment
from signalai.utils.logging import setup_logger

def main():
    parser = argparse.ArgumentParser(description="Run vibration classification experiment.")
    parser.add_argument("classifier", choices=["svm", "rf", "lgbm"], help="Classifier to use")
    parser.add_argument("dataset", help="Dataset name (e.g., MFPT, CWRU_12K)")
    parser.add_argument("transform", help="Feature transform name (e.g., time, frequency)")
    parser.add_argument("--selector", choices=["sfs", "anova", "hybrid", "mrmr"], help="Feature selector")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--no-timestamp", action="store_true", help="Do not create a timestamped subfolder")
    
    args = parser.parse_args()

    # Pre-calculate experiment name and timestamp to setup logging early
    exp_name = f"Exp_{args.classifier}_{args.dataset}_{args.transform}"
    if args.selector:
        exp_name += f"_{args.selector}"

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    if args.no_timestamp:
        run_dir = args.output
    else:
        run_dir = os.path.join(args.output, f"results_{exp_name}_{timestamp}")
    
    os.makedirs(run_dir, exist_ok=True)
    
    log_file = os.path.join(run_dir, "experiment.log")
    setup_logger(log_file)

    print("=== Running experiment ===")
    print(f"Dataset: {args.dataset}")
    print(f"Classifier: {args.classifier}")
    print(f"Transform: {args.transform}")
    if args.selector:
        print(f"Selector: {args.selector}")
    print(f"Output directory: {run_dir}")
    print("==========================")

    # --- Dataset Setup ---
    dataset_key = args.dataset.split("_")[0]
    raw_root_dir = f"./data/raw_data/{dataset_key}"
    deep_root_dir = f"./data/deep_data/{args.dataset}_{args.transform}"

    raw_dataset_fn = getattr(raw_datasets, f"{dataset_key.upper()}_raw")
    raw_dataset = raw_dataset_fn(raw_root_dir, download=True)
    print(f"Raw dataset loaded with length: {len(raw_dataset)}")

    # Filtering logic
    filter_obj = None
    if "CWRU" in args.dataset:
        sr = 48000 if "48K" in args.dataset else 12000
        filter_obj = FilterByValue(on_field="sample_rate", values=sr)
    elif "MFPT" in args.dataset:
        filter_obj = FilterByValue(on_field="sample_rate", values=48828)

    # Get pipeline
    pipeline = get_pipeline(args.transform)
    
    # Convert/Load Deep Dataset
    print(f"Converting dataset...")
    deep_dataset = convertDataset(
        raw_dataset, filter=filter_obj, transforms=pipeline,
        dir_path=deep_root_dir, batch_size=16
    )
    print(f"Dataset converted and has length: {len(deep_dataset)}")
    
    feature_names = get_feature_names(args.transform)
    print(f"Number of features extracted: {len(feature_names)}")

    # --- Fold Generation ---
    print("Generating folds...")
    GroupClass, deep_dataset = get_dataset_grouping(args.dataset, deep_dataset)

    generator = FoldIdxGeneratorUnbiased(
        deep_dataset, GroupClass,
        dataset_name=args.dataset
    )
    folds = generator.generate_folds()
    print("Folds generated.\n")

    # --- Classifier & Selector ---
    if args.classifier == "svm":
        model = SVC(random_state=42, probability=True)
        search_space = {
            "model__C": [0.1, 1, 10, 100],
            "model__kernel": ["linear", "rbf", "poly"],
            "model__gamma": ["scale", "auto"]
        }
    elif args.classifier == "lgbm":
        # n_jobs=1: GridSearchCV(n_jobs=-1) already parallelizes across fits/candidates;
        # without this, LightGBM's own internal OpenMP threading oversubscribes the CPU
        # (N worker processes x M internal threads each, contending for the same cores).
        # subsample_freq=1 is fixed (not searched) purely so the "model__subsample" grid
        # value below actually takes effect: LightGBM's row-bagging is a no-op unless
        # subsample_freq > 0.
        # min_child_samples default (20) requires >=40 samples in a node to split at
        # all, which is larger than some datasets' entire inner-CV training folds (e.g.
        # MFPT: ~27 samples/fold), silently collapsing every tree to a single constant
        # leaf. Fixed at a low value rather than grid-searched to keep the search space
        # (and runtime) unchanged.
        model = LGBMClassifier(random_state=42, verbose=-1, n_jobs=1, subsample_freq=1, min_child_samples=5)
        # Dimensions follow Probst et al. (2019, JMLR "Tunability"), which studied this
        # same leaf-wise boosted-tree family (XGBoost/gbm) and ranked learning_rate,
        # n_estimators, tree-complexity (max_depth/num_leaves), subsample (row bagging),
        # and colsample_bytree (feature bagging) as the highest-impact hyperparameters.
        # num_leaves is capped at 63 and subsample/colsample_bytree add bagging-based
        # regularization, since the per-fold sample counts here are small enough that an
        # unconstrained leaf-wise learner overfits easily.
        search_space = {
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__n_estimators": [100, 300, 500],
            "model__num_leaves": [15, 31, 63],
            "model__colsample_bytree": [0.7, 1.0],
            "model__subsample": [0.7, 1.0],
        }
    else:
        model = RandomForestClassifier(random_state=42)
        search_space = {
            "model__n_estimators": [50, 100, 200],
            "model__criterion": ["gini", "entropy", "log_loss"],
            "model__max_depth": [10, 25, 50],
            "model__min_samples_split": [2, 5, 10]
        }

    selector = None
    if args.selector == "sfs":
        selector = SequentialFeatureSelector(KNeighborsClassifier(n_neighbors=3), n_features_to_select=20)
    elif args.selector == "anova":
        selector = SelectKBest(f_classif, k=20)
    elif args.selector == "hybrid":
        if args.classifier == "svm":
            hrw_model = SVC(kernel="linear", C=1.0, random_state=42)
        else:
            hrw_model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
        selector = HybridRankingWrapperSelector(base_estimator=hrw_model, max_features_ratio=0.2)
    elif args.selector == "mrmr":
        selector = MRMRSelector(n_features_to_select=20, verbose=True)

    # --- Run Experiment ---
    experiment = ClassificationExperiment(
        name=exp_name,
        description=f"Refactored experiment on {args.dataset}",
        dataset=deep_dataset,
        data_fold_idxs=folds,
        feature_names=feature_names,
        feature_selector=selector,
        model=model,
        model_parameters_search_space=search_space,
        output_dir=args.output,
        start_time=timestamp if not args.no_timestamp else ""
    )

    experiment.run()

if __name__ == "__main__":
    main()
