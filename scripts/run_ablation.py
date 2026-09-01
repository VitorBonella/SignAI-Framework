"""
Feature-representation ablation study: leave-one-feature-group-out of the "all"
pipeline, run with a single classifier (set via CLASSIFIER_NAME: "rf" or "lgbm")
across all datasets.

This script is also an example of composing a *new* feature pipeline directly out
of signalai's building blocks (Transform classes, Sequential, Aggregator,
FeatureExtractor) instead of using one of the pre-built pipelines from
signalai.features.pipelines.get_pipeline() -- useful as a template for anyone
extending the framework with their own feature representations.
"""
import os
import time

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

import vibdata.raw as raw_datasets
from vibdata.deep.DeepDataset import convertDataset
from vibdata.deep.signal.transforms import (
    FilterByValue, Sequential, Aggregator, FeatureExtractor, SplitSampleRate, FFT
)

from signalai.features.pipelines import (
    FEATURES_TIME, FEATURES_FREQ, FEATURES_WAVELET, FEATURES_PSD, FEATURES_SPECTRAL
)
from signalai.features.wavelet import LevelCorrelationCoefficients, RelativeEnergyRatio
from signalai.features.custom import WaveletCoeffsFeatures, PowerSpectralDensity, SpectralEnvelope

from signalai.data.grouping import get_dataset_grouping
from signalai.sampling.generators import FoldIdxGeneratorUnbiased
from signalai.experiments.classification import ClassificationExperiment
from signalai.utils.logging import setup_logger


def _wavelet_branch():
    branch = WaveletCoeffsFeatures(features=FEATURES_WAVELET)
    branch.wavelet_multilevel_feat = [LevelCorrelationCoefficients(), RelativeEnergyRatio()]
    return branch

def _wavelet_feature_names():
    names = []
    for f in FEATURES_WAVELET:
        for level in range(1, 6): # 4 detail levels + 1 approximation
            names.append(f"{type(f).__name__}_wavelet_L{level}")
    return names + ["LevelCorrelationCoefficients_wavelet", "RelativeEnergyRatio_wavelet"]

# The 5 feature groups that make up the "all" pipeline (mirrors PIPELINE_ALL in
# signalai/features/pipelines.py), as (transform_branch, feature_names) pairs. Building
# every ablation from this single mapping guarantees each "all_minus_X" pipeline
# stays a strict subset of "all" -- no group can be duplicated or omitted by accident.
FEATURE_GROUPS = {
    "time": (
        FeatureExtractor(features=FEATURES_TIME),
        [type(f).__name__ + "_time" for f in FEATURES_TIME],
    ),
    "frequency": (
        Sequential([FFT(), FeatureExtractor(features=FEATURES_FREQ)]),
        [type(f).__name__ + "_freq" for f in FEATURES_FREQ],
    ),
    "wavelet": (
        _wavelet_branch(),
        _wavelet_feature_names(),
    ),
    "psd": (
        Sequential([PowerSpectralDensity(), FeatureExtractor(features=FEATURES_PSD)]),
        [type(f).__name__ + "_psd" for f in FEATURES_PSD],
    ),
    "spectral_envelope": (
        Sequential([SpectralEnvelope(n_lpc=16), FeatureExtractor(features=FEATURES_SPECTRAL)]),
        [type(f).__name__ + "_spectral" for f in FEATURES_SPECTRAL],
    ),
}

def build_ablation_pipeline(exclude_group):
    branches = [branch for key, (branch, _) in FEATURE_GROUPS.items() if key != exclude_group]
    return Sequential([SplitSampleRate(), Aggregator(branches)])

def build_ablation_feature_names(exclude_group):
    names = []
    for key, (_, group_names) in FEATURE_GROUPS.items():
        if key != exclude_group:
            names += group_names
    return names

def get_model_and_search_space(classifier_name):
    """Model + search space per classifier, matching scripts/run_classification.py."""
    if classifier_name == "lgbm":
        model = LGBMClassifier(random_state=42, verbose=-1, n_jobs=1, subsample_freq=1, min_child_samples=5)
        search_space = {
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__n_estimators": [100, 300, 500],
            "model__num_leaves": [15, 31, 63],
            "model__colsample_bytree": [0.7, 1.0],
            "model__subsample": [0.7, 1.0],
        }
    elif classifier_name == "rf":
        model = RandomForestClassifier(random_state=42)
        search_space = {
            "model__n_estimators": [50, 100, 200],
            "model__criterion": ["gini", "entropy", "log_loss"],
            "model__max_depth": [10, 25, 50],
            "model__min_samples_split": [2, 5, 10]
        }
    else:
        raise ValueError(f"Unsupported classifier for ablation: {classifier_name}")
    return model, search_space

# --- Ablation configuration ---
DATASETS = ["CWRU_12K", "CWRU_48K", "UOC", "PU", "IMS", "MFPT"]
ABLATED_GROUPS = ["time", "frequency", "wavelet", "psd", "spectral_envelope"]
CLASSIFIER_NAME = "rf"
BASE_RESULTS_DIR = "results"


def run_one(dataset, exclude_group):
    transform_name = f"all_minus_{exclude_group}"
    exp_name = f"Exp_{CLASSIFIER_NAME}_{dataset}_{transform_name}"

    run_dir = os.path.join(BASE_RESULTS_DIR, dataset, transform_name, CLASSIFIER_NAME)
    os.makedirs(run_dir, exist_ok=True)
    setup_logger(os.path.join(run_dir, "experiment.log"))

    print("=== Running ablation experiment ===")
    print(f"Dataset: {dataset}")
    print(f"Classifier: {CLASSIFIER_NAME}")
    print(f"Ablation: leaving out '{exclude_group}'")
    print(f"Output directory: {run_dir}")
    print("==========================")

    # --- Dataset Setup (same conventions as scripts/run_classification.py) ---
    dataset_key = dataset.split("_")[0]
    raw_root_dir = f"./data/raw_data/{dataset_key}"
    deep_root_dir = f"./data/deep_data/{dataset}_{transform_name}"

    raw_dataset_fn = getattr(raw_datasets, f"{dataset_key.upper()}_raw")
    raw_dataset = raw_dataset_fn(raw_root_dir, download=True)
    print(f"Raw dataset loaded with length: {len(raw_dataset)}")

    filter_obj = None
    if "CWRU" in dataset:
        sr = 48000 if "48K" in dataset else 12000
        filter_obj = FilterByValue(on_field="sample_rate", values=sr)
    elif "MFPT" in dataset:
        filter_obj = FilterByValue(on_field="sample_rate", values=48828)

    pipeline = build_ablation_pipeline(exclude_group)
    feature_names = build_ablation_feature_names(exclude_group)

    print("Converting dataset...")
    deep_dataset = convertDataset(
        raw_dataset, filter=filter_obj, transforms=pipeline,
        dir_path=deep_root_dir, batch_size=16
    )
    print(f"Dataset converted and has length: {len(deep_dataset)}")
    print(f"Number of features extracted: {len(feature_names)}")

    # --- Fold Generation ---
    print("Generating folds...")
    GroupClass, deep_dataset = get_dataset_grouping(dataset, deep_dataset)
    generator = FoldIdxGeneratorUnbiased(deep_dataset, GroupClass, dataset_name=dataset)
    folds = generator.generate_folds()
    print("Folds generated.\n")

    # --- Classifier (model + search space match scripts/run_classification.py) ---
    model, search_space = get_model_and_search_space(CLASSIFIER_NAME)

    # --- Run Experiment ---
    experiment = ClassificationExperiment(
        name=exp_name,
        description=f"Feature-representation ablation (leave out '{exclude_group}') on {dataset}",
        dataset=deep_dataset,
        data_fold_idxs=folds,
        feature_names=feature_names,
        feature_selector=None,
        model=model,
        model_parameters_search_space=search_space,
        output_dir=run_dir,
        start_time=""
    )
    experiment.run()


def main():
    total_start_time = time.time()
    total_experiments = len(DATASETS) * len(ABLATED_GROUPS)
    current_exp_count = 0

    print("=" * 80)
    print(f"STARTING ABLATION EXPERIMENTS ({total_experiments} total)")
    print(f"Classifier: {CLASSIFIER_NAME}")
    print(f"Ablated groups: {ABLATED_GROUPS}")
    print("=" * 80)

    for dataset in DATASETS:
        for exclude_group in ABLATED_GROUPS:
            current_exp_count += 1
            print(f"\n[{current_exp_count}/{total_experiments}] "
                  f"DS: {dataset} | CLF: {CLASSIFIER_NAME} | Excluding: {exclude_group}")

            start = time.time()
            try:
                run_one(dataset, exclude_group)
                print(f"✅ Success! Duration: {time.time() - start:.2f}s")
            except Exception as e:
                print(f"❌ Failed: {e}. Duration: {time.time() - start:.2f}s")

            elapsed_time = time.time() - total_start_time
            avg_time_per_exp = elapsed_time / current_exp_count
            remaining_exps = total_experiments - current_exp_count
            estimated_remaining_time = avg_time_per_exp * remaining_exps
            print(f"Elapsed: {elapsed_time/60:.2f}m | Estimated Remaining: {estimated_remaining_time/60:.2f}m")
            print("-" * 40)

    total_duration = time.time() - total_start_time
    print("=" * 80)
    print("ALL ABLATION EXPERIMENTS COMPLETED.")
    print(f"Total Duration: {total_duration/3600:.2f} hours")
    print("=" * 80)

if __name__ == "__main__":
    main()
