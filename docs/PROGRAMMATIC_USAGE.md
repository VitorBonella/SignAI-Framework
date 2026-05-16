# Programmatic Usage & Custom Benchmarks

The `scripts/run_classification.py` file provided in the repository is merely an **example** of how to execute a standard experiment. 

The true purpose and power of the **signalAI** framework lie in its **programmatic API**. It acts as a standardized benchmarking methodology. By writing your own Python scripts and importing `signalAI` classes, you can rigorously benchmark custom vibration-signal processing pipelines, newly engineered features, novel Deep Learning architectures, or custom datasets—all while automatically adhering to the framework's strict rules for fair sampling (preventing data leakage) and reproducible cross-validation.

This guide will show you how to build custom benchmarks programmatically.

---

## 1. Benchmarking Custom Feature Pipelines (Classical ML)

If your research proposes a new way to process signals or extract features (e.g., a novel mathematical transform), you can benchmark it against standard methods. 

Instead of modifying the framework, you create a custom `vibdata` pipeline in your code and feed it into `ClassificationExperiment`.

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from vibdata.deep.signal.transforms import Transform, FeatureExtractor, Sequential, SplitSampleRate
from signalai.experiments.classification import ClassificationExperiment
from signalai.data.grouping import get_dataset_grouping
from signalai.sampling.generators import FoldIdxGeneratorUnbiased

# --- 1. Define your custom transform in your own code ---
class MyCustomTransform(Transform):
    def apply(self, data):
        # Example: Square the signal or apply a custom filter
        return np.square(data)

# --- 2. Build your custom pipeline and feature list ---
from vibdata.deep.signal.transforms import Kurtosis, RootMeanSquare, Mean
from signalai.features.pipelines import FEATURES_TIME

# You can use the standard framework features, modify them, or create your own list:
custom_features = [
    Kurtosis(),
    RootMeanSquare(),
    Mean()
    # You could also append framework defaults: custom_features + FEATURES_TIME
]

my_pipeline = Sequential([
    SplitSampleRate(),
    MyCustomTransform(),
    FeatureExtractor(features=custom_features) # Extract your custom set of features
])

# --- 3. Load dataset and group it ---
# (Assume `raw_dataset` is loaded via vibdata)
from vibdata.deep.DeepDataset import convertDataset
deep_dataset = convertDataset(raw_dataset, transforms=my_pipeline, dir_path="./my_data_cache")

GroupClass, grouped_dataset = get_dataset_grouping("CWRU_12K", deep_dataset)

# --- 4. Generate Unbiased Folds ---
generator = FoldIdxGeneratorUnbiased(grouped_dataset, GroupClass)
folds = generator.generate_folds()

# --- 5. Run Benchmark ---
experiment = ClassificationExperiment(
    name="MyCustomPipelineBenchmark",
    description="Benchmarking MyCustomTransform with Random Forest",
    dataset=grouped_dataset,
    data_fold_idxs=folds,
    feature_names=["Kurtosis", "RootMeanSquare", "Mean"], # Match your custom features
    model=RandomForestClassifier(),
    model_parameters_search_space={"model__n_estimators": [50]}
)

results = experiment.run()
```

---

## 2. Benchmarking Deep Learning Architectures

For Deep Learning, your benchmark often focuses on the **network architecture** (e.g., comparing a 1D-CNN vs. an LSTM) operating on raw or minimally processed signals (like STFT spectrograms).

You can use the `DeepLearningExperiment` (which utilizes PyTorch) to ensure your neural networks are evaluated under the exact same unbiased cross-validation methodology as classical ML models.

```python
import torch.nn as nn
from signalai.experiments.deep_learning import DeepLearningExperiment
from signalai.data.grouping import get_dataset_grouping
from signalai.sampling.generators import FoldIdxGeneratorUnbiased

# --- 1. Define your custom PyTorch architecture ---
class My1DCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=64, stride=8),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.fc = nn.Linear(16 * 127, num_classes) # Example dimensions

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# --- 2. Load dataset and generate folds ---
# (Load your deep_dataset here containing raw time-series)
GroupClass, grouped_dataset = get_dataset_grouping("MFPT", deep_dataset)
generator = FoldIdxGeneratorUnbiased(grouped_dataset, GroupClass)
folds = generator.generate_folds()

# --- 3. Run Benchmark ---
experiment = DeepLearningExperiment(
    name="CNN_Benchmark",
    description="Benchmarking custom 1D CNN on MFPT",
    dataset=grouped_dataset,
    data_fold_idxs=folds,
    model=My1DCNN(num_classes=3),
    batch_size=32,
    lr=0.001,
    num_epochs=50,
    device="cuda"
)

results = experiment.run()
```

---

## 3. Benchmarking Custom Datasets

If you have collected a novel proprietary dataset and want to benchmark models on it, you must ensure fair sampling. You do this by creating a custom grouping strategy.

```python
from signalai.data.grouping import GroupDataset

# --- 1. Define Custom Grouping ---
class GroupMyCustomDataset(GroupDataset):
    """
    Groups samples together by their physical operating condition (e.g., 'Load' and 'RPM').
    This ensures samples from the same physical setup aren't split across train/test, preventing leakage.
    """
    def _create_groups(self):
        # Implement logic to group your deep_dataset.
        # This will depend on the metainfo structure of your vibdata implementation.
        pass

# --- 2. Generate folds using your new grouping ---
generator = FoldIdxGeneratorUnbiased(my_proprietary_dataset, GroupMyCustomDataset)
folds = generator.generate_folds()

# Proceed to run standard or custom experiments...
```

## Summary
By using the framework programmatically:
1. **Pipelines / Features:** You can benchmark the impact of new signal processing techniques.
2. **Models / Architectures:** You can benchmark the predictive power of new ML algorithms or deep neural networks.
3. **Data:** You can rigorously evaluate existing algorithms on newly recorded datasets.

All while relying on `signalAI` to enforce strict academic rigor through fair sampling, zero data-leakage cross-validation, and standardized JSON result tracking.
