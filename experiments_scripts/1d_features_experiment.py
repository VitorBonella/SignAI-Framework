import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import vibdata.raw as raw_datasets
from vibdata.deep.signal.transforms import (
    Kurtosis, RootMeanSquare, StandardDeviation, Mean, LogAttackTime,
    TemporalDecrease, TemporalCentroid, EffectiveDuration, ZeroCrossingRate,
    PeakValue, CrestFactor, Skewness, ClearanceFactor, ImpulseFactor,
    ShapeFactor, UpperBoundValueHistogram, LowerBoundValueHistogram,
    Variance, PeakToPeak, Transform, Sequential, SplitSampleRate,
    FeatureExtractor, FilterByValue, Aggregator, FFT
)
from vibdata.deep.DeepDataset import convertDataset
from vibdata.deep.signal.core import SignalSample
from signalAI.utils.group_dataset import GroupDataset
from signalAI.utils.fold_idx_generator import FoldIdxGeneratorUnbiased
from signalAI.experiments.features_1d import Features1DExperiment


# ======================================
# Feature extraction setup
# ======================================
features_funcs = [
    Kurtosis(), RootMeanSquare(), StandardDeviation(), Mean(), LogAttackTime(),
    TemporalDecrease(), TemporalCentroid(), EffectiveDuration(), ZeroCrossingRate(),
    PeakValue(), CrestFactor(), Skewness(), ClearanceFactor(), ImpulseFactor(),
    ShapeFactor(), UpperBoundValueHistogram(), LowerBoundValueHistogram(), Variance(), PeakToPeak()
]

transforms_time = Sequential([
    SplitSampleRate(),
    FeatureExtractor(features=features_funcs),
])

transforms_frequency = Sequential([
    SplitSampleRate(),
    FFT(),
    FeatureExtractor(features=features_funcs),
])

transforms_time_frequency = Sequential([
    SplitSampleRate(),
    Aggregator([
        FeatureExtractor(features=features_funcs),  # Time domain features
        Sequential([FFT(), FeatureExtractor(features=features_funcs)])  # Frequency domain features
    ])
])


# ======================================
# Dataset grouping strategies
# ======================================
class GroupMultiRoundMFPT(GroupDataset):
    @staticmethod
    def _assigne_group(sample: SignalSample) -> int:
        sample_metainfo = sample["metainfo"]
        return sample_metainfo["label"].astype(str) + " " + sample_metainfo["load"].astype(int).astype(str)


class GroupMultiRoundCWRULoad(GroupDataset):
    @staticmethod
    def _assigne_group(sample: SignalSample) -> int:
        sample_metainfo = sample["metainfo"]
        return sample_metainfo["label"].astype(str) + " " + sample_metainfo["load"].astype(int).astype(str)


# ======================================
# Main experiment runner
# ======================================
def main(classifier_name, dataset_name, transform_name, transforms):
    print(f"\n=== Running experiment ===")
    print(f"Dataset: {dataset_name}")
    print(f"Classifier: {classifier_name}")
    print(f"Transform: {transform_name}")
    print("==========================")

    # ---- Dataset setup ----
    dataset_key = dataset_name.split("_")[0]
    raw_root_dir = f"../data/raw_data/{dataset_key}_{transform_name}"
    deep_root_dir = f"../data/deep_data/{dataset_key}_{transform_name}"

    raw_dataset_fn = getattr(raw_datasets, dataset_key + "_raw")
    raw_dataset = raw_dataset_fn(raw_root_dir, download=True)
    print("Raw dataset loaded with length:", len(raw_dataset))

    # ---- Filtering ----
    if "CWRU" in dataset_name:
        if "48k" in dataset_name:
            filter = FilterByValue(on_field="sample_rate", values=48000)
        elif "12k" in dataset_name:
            filter = FilterByValue(on_field="sample_rate", values=12000)
        else:
            filter = None
    elif "MFPT" in dataset_name:
        filter = FilterByValue(on_field="sample_rate", values=48828)
    else:
        filter = None

    # ---- Convert dataset ----
    print("Converting dataset...")
    deep_dataset = convertDataset(raw_dataset, filter=filter, transforms=transforms,
                                  dir_path=deep_root_dir, batch_size=32)
    print("Dataset converted and has length:", len(deep_dataset))

    # ---- Fold generation ----
    print("Generating folds...")
    if "MFPT" in dataset_name:
        CLASS_DEF = {23: "N", 25: "O", 24: "I"}
        
        class foo(dict):
            def __getitem__(self, key):
                return str(key)

        CONDITION_DEF = foo()
        GroupClass = GroupMultiRoundMFPT
    else:
        CLASS_DEF = {0: "N", 1: "O", 2: "I", 3: "R"}
        CONDITION_DEF = {"0": "0", "1": "1", "2": "2", "3": "3"}
        GroupClass = GroupMultiRoundCWRULoad

    folds_multiround = FoldIdxGeneratorUnbiased(
        deep_dataset,
        GroupClass,
        dataset_name=dataset_name + "_" + transform_name,
        multiround=True,
        class_def=CLASS_DEF,
        condition_def=CONDITION_DEF
    ).generate_folds()
    print("Folds generated.")

    # ---- Classifier setup ----
    if classifier_name == "svm":
        from sklearn.svm import SVC
        model = SVC(random_state=42)
        model_parameters_search_space = {
            "model__C": [0.1, 1, 10, 100],
            "model__kernel": ["linear", "rbf", "poly"],
            "model__gamma": ["scale", "auto"]
        }
    elif classifier_name == "rf":
        model = RandomForestClassifier(random_state=42)
        model_parameters_search_space = {
            "model__n_estimators": [50, 100, 200],
            "model__criterion": ["gini", "entropy", "log_loss"],
            "model__max_depth": [10, 25, 50],
            "model__min_samples_split": [2, 5, 10]
        }

    # ---- Experiment ----
    experiment = Features1DExperiment(
        name=f"Vibration_Analysis_{classifier_name.upper()}_{dataset_name}_{transform_name}",
        description="Feature extraction and classification on vibration datasets",
        feature_names=features_funcs,
        dataset=deep_dataset,
        data_fold_idxs=folds_multiround,
        n_inner_folds=4,
        model=model,
        model_parameters_search_space=model_parameters_search_space
    )

    experiment.run()


# ======================================
# Script entrypoint
# ======================================
if __name__ == "__main__":
    valid_classifiers = ["svm", "rf"]
    valid_datasets = ["MFPT", "CWRU12k", "CWRU48k"]
    valid_transforms = ["time", "frequency", "time_and_frequency"]

    if len(sys.argv) < 4:
        raise ValueError("Usage: python script.py <classifier> <dataset> <transform>")

    classifier = sys.argv[1]
    dataset = sys.argv[2].upper()
    transform_name = sys.argv[3]

    if classifier not in valid_classifiers:
        raise ValueError(f"Classifier {classifier} not recognized. Valid options: {valid_classifiers}")

    if dataset not in valid_datasets:
        raise ValueError(f"Dataset {dataset} not recognized. Valid options: {valid_datasets}")

    if transform_name not in valid_transforms:
        raise ValueError(f"Transform {transform_name} not recognized. Valid options: {valid_transforms}")

    # ---- Select transform ----
    if transform_name == "time":
        transforms = transforms_time
    elif transform_name == "frequency":
        transforms = transforms_frequency
    else:
        transforms = transforms_time_frequency

    main(classifier, dataset, transform_name, transforms)
