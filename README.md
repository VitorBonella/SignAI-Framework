# signalAI: Vibration-Signal Fault Diagnosis Benchmark Framework

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/signalai.svg)](https://badge.fury.io/py/signalai)

**signalAI** is a modular, extensible, and highly reproducible Python framework developed as part of a Master's Thesis. It is designed to benchmark vibration-signal fault diagnosis methods using fair, robust, and reproducible sampling strategies across the most popular public datasets.

This framework standardizes the evaluation of machine learning and deep learning models by addressing common pitfalls in vibration signal analysis, such as data leakage and biased cross-validation folds. It integrates seamlessly with the `vibdata` library for raw data loading and transformations.

---

## Key Features

- **Fair Benchmarking**: Implements rigorous grouping strategies (e.g., ensuring samples from the same physical condition/load are kept together) to prevent data leakage during cross-validation.
- **Unbiased Fold Generation**: Advanced multi-round sampling to guarantee fair comparisons between models.
- **Modular Pipeline**: Decoupled components for data loading, feature extraction, fold generation, and experiment execution.
- **High Extensibility**: Easily use framework classes in your own codebase to define custom datasets, preprocessing pipelines, or entirely new experimental setups without modifying the framework's source code.
- **Unified Evaluation**: Standardized metrics and result serialization for transparent and reproducible academic reporting.

---

## Architecture Overview

The framework is structured into a clean `signalai` Python package:

- `signalai.core`: Base abstractions (`BaseExperiment`) that define the lifecycle of any benchmarking experiment.
- `signalai.data`: Grouping strategies tailored for popular datasets (MFPT, CWRU, PU, IMS, UOC) to ensure fair data partitioning.
- `signalai.features`: Standardized feature extraction pipelines (Time, Frequency, Wavelet, PSD, etc.) and modular transforms.
- `signalai.sampling`: Advanced fold generators (`FoldIdxGeneratorUnbiased`) for rigorous cross-validation setups.
- `signalai.experiments`: Ready-to-use experiment runners for classical Machine Learning (`ClassificationExperiment`) and Deep Learning.
- `signalai.utils`: Centralized logging, metrics calculation, and JSON-based result serialization.

---

## Installation

We recommend installing the framework in a dedicated virtual environment.

You can install the latest release directly from PyPI:

```bash
pip install signalai
```

Alternatively, you can install the latest development version from the repository:

```bash
# Clone the repository
git clone https://github.com/VitorBonella/SignAI-Framework.git
cd SignAI-Framework

# Install dependencies and the framework in editable mode
pip install -e .
```

*Note: Ensure you have the `vibdata` library installed or accessible in your environment, as it is a core dependency for raw dataset handling.*

---

## Usage Guide

### 1. Command-Line Interface (CLI) - *Example Usage*

The framework includes an example script to demonstrate how benchmarking works. *Note: The true power of the framework lies in programmatic usage (see below), but you can use this script to test standard pipelines:*

```bash
python scripts/run_classification.py rf CWRU_12K time --output ./my_results
```

**Available CLI Options:**
- **Classifiers:** `rf` (Random Forest), `svm` (Support Vector Machine)
- **Datasets:** `MFPT`, `CWRU_12K`, `CWRU_48K`, `PU`, `IMS`, `UOC`
- **Transforms:** `time`, `frequency`, `time_and_frequency`, `wavelet`, `psd`, `spectral_envelope`, `all`
- **Feature Selectors:** `--selector sfs` or `--selector anova`

### 2. Programmatic Usage & Custom Benchmarks (Primary Use Case)

The CLI scripts are just examples. **The correct way to use the framework is programmatically.**

By writing your own Python scripts and importing `signalAI` classes, you can rigorously benchmark custom vibration-signal processing pipelines, newly engineered features, novel Deep Learning architectures, or custom datasets. The framework enforces a standard methodology (fair sampling, avoiding leakage) regardless of the components you are benchmarking.

👉 **[Read the comprehensive guide on Programmatic Usage & Custom Benchmarks](./docs/PROGRAMMATIC_USAGE.md)** to learn how to:
- Benchmark custom Preprocessing and Feature Pipelines.
- Benchmark custom Deep Learning Network Architectures (PyTorch).
- Benchmark proprietary datasets.

**Brief Example (Classical ML):**

```python
from sklearn.ensemble import RandomForestClassifier
from signalai.features.pipelines import get_pipeline, get_feature_names
from signalai.experiments.classification import ClassificationExperiment
from signalai.data.grouping import get_dataset_grouping
from signalai.sampling.generators import FoldIdxGeneratorUnbiased

# 1. Setup Feature Pipeline
transform_name = "time"
pipeline = get_pipeline(transform_name)
feature_names = get_feature_names(transform_name)

# 2. Load your deep_dataset using vibdata 
# (Assuming `deep_dataset` is instantiated here)

# 3. Apply Grouping Strategy and Generate Unbiased Folds
GroupClass, grouped_dataset = get_dataset_grouping("CWRU_12K", deep_dataset)
generator = FoldIdxGeneratorUnbiased(grouped_dataset, GroupClass)
folds = generator.generate_folds()

# 4. Define and Run the Experiment
experiment = ClassificationExperiment(
    name="MyCustomExperiment",
    description="Evaluating RF with custom unbiased folds",
    dataset=grouped_dataset,
    data_fold_idxs=folds,
    feature_names=feature_names,
    model=RandomForestClassifier(random_state=42),
    model_parameters_search_space={"model__n_estimators": [50, 100]}
)

# Executes the pipeline, cross-validation, and saves metrics
experiment.run()
```

---

## Extending and Customizing

The true power of `signalAI` lies in its extensibility. You can extend the framework by subclassing its core components in your own external code. **You do not need to modify the framework's internal files.**

### Custom Datasets & Grouping
To introduce a new dataset or a custom sampling strategy, create a subclass of the grouping base classes in your own script. This ensures that samples are grouped correctly to avoid data leakage before passing them to the fold generator.

### Custom Feature Transforms
You can create new feature extraction steps by subclassing the `Transform` class (from `vibdata` or `signalai`). Simply define the `apply()` logic and inject your custom transform into a pipeline before creating the `ClassificationExperiment`.

### Custom Experiments
If the default `ClassificationExperiment` does not fit your needs (e.g., you are building an autoencoder or a specific deep learning architecture), you can subclass `BaseExperiment`:

```python
from signalai.core.experiment import BaseExperiment

class MyDeepLearningExperiment(BaseExperiment):
    def prepare_data(self):
        # Custom logic for DataLoader setup
        pass
        
    def run(self):
        # Custom training loop
        self.prepare_data()
        # Train, validate, calculate metrics...
        self.save_results(self.run_dir / "results.json")
```

---

## Results and Reproducibility

Every experiment run automatically generates a time-stamped directory (e.g., `results_Exp_rf_CWRU_12K_time_20260516_112536`) containing:
- `experiment.log`: A detailed trace of the execution, data shapes, and metrics.
- `results.json` (or similar outputs): Standardized tracking of cross-validation accuracy, F1-scores, feature selection states, and hyperparameter setups.

This guarantees that every result reported in your thesis or paper can be identically reproduced.

---

## Academic Context & Citation

This framework was developed as the core software artifact for a Master's Thesis focusing on the rigor and reproducibility of vibration-signal fault diagnosis methods. 

If you use this framework in your research, please consider citing it.
