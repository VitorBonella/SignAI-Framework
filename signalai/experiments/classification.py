import numpy as np
import time
import os
from pathlib import Path
from typing import Any, List, Callable, Dict, Optional, Tuple, Union

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from vibdata.deep.DeepDataset import DeepDataset

from signalai.core.experiment import BaseExperiment
from signalai.utils.metrics import calculate_metrics
from signalai.utils.results import ExperimentResults, FoldResults

class ClassificationExperiment(BaseExperiment):
    """
    Standard experiment for signal classification using extracted features.
    Supports single and multi-round cross-validation.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        dataset: DeepDataset = None,
        data_fold_idxs: Union[np.ndarray, List[np.ndarray]] = None,
        feature_names: List[str] = None,
        n_inner_folds: int = 4,
        feature_selector: Optional[Any] = None,
        output_dir: str = "results",
        random_state: int = 42,
        model_parameters_search_space: Optional[Dict[str, Any]] = None,
        scaler: Optional[Any] = StandardScaler,
        **kwargs
    ):
        super().__init__(name, description, dataset, output_dir=output_dir, **kwargs)
        self.n_inner_folds = n_inner_folds
        self.feature_names = feature_names
        self.feature_selector = feature_selector
        self.random_state = random_state
        self.data_fold_idxs = data_fold_idxs
        self.model_parameters_search_space = model_parameters_search_space
        self.scaler = scaler if scaler is not None else StandardScaler
        
        # Determine number of outer folds
        if isinstance(data_fold_idxs, list):
            self.n_outer_folds = len(np.unique(data_fold_idxs[0]))
        else:
            self.n_outer_folds = len(np.unique(data_fold_idxs))
        
        self.prepare_data()

    def prepare_data(self):
        """Extracts features and labels from the DeepDataset."""
        features, labels = [], []
        for sample in self.dataset:
            # We assume the signal contains the extracted features (1D array)
            features.append(sample['signal'][0])
            labels.append(sample['metainfo']['label'])
        
        self.X = np.array(features)
        self.y = np.array(labels)

    def _create_pipeline(self) -> Pipeline:
        """Creates the sklearn pipeline."""
        steps = [('scaler', self.scaler())]
        if self.feature_selector is not None:
            steps.append(('feature_selector', self.feature_selector))
        
        if self.model is None:
            raise ValueError("Model not defined for the experiment")
            
        steps.append(('model', self.model))
        return Pipeline(steps)

    def _run_inner_cv(self, X_train: np.ndarray, y_train: np.ndarray):
        """Executes inner CV for hyperparameter tuning."""
        inner_cv = StratifiedKFold(
            n_splits=self.n_inner_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        pipeline = self._create_pipeline()
        search = GridSearchCV(
            pipeline,
            self.model_parameters_search_space,
            cv=inner_cv,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=1
        )
        
        search.fit(X_train, y_train)
        
        selected_features = None
        if 'feature_selector' in search.best_estimator_.named_steps:
            selector = search.best_estimator_.named_steps['feature_selector']
            if hasattr(selector, 'get_support'):
                selected_features = [
                    self.feature_names[i] for i in selector.get_support(indices=True)
                ]

        return search.best_estimator_, search.best_params_, selected_features

    def run_single_round(self, round_idx: int = None) -> ExperimentResults:
        """Runs a single round of nested cross-validation."""
        start_time_round = time.time()
        
        X, y = self.X, self.y
        folds = self.data_fold_idxs[round_idx] if round_idx is not None else self.data_fold_idxs

        results = ExperimentResults(
            experiment_name=self.name,
            description=self.description,
            model_name=self.model.__class__.__name__,
            feature_names=self.feature_names,
            config={
                'n_outer_folds': self.n_outer_folds,
                'n_inner_folds': self.n_inner_folds,
                'random_state': self.random_state,
                'round': round_idx
            }
        )
        
        for outer_fold in range(self.n_outer_folds):
            print(f"\n=== Outer Fold {outer_fold + 1}/{self.n_outer_folds} ===")
            
            train_mask = folds != outer_fold
            test_mask = folds == outer_fold
            
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
            
            best_pipeline, best_params, selected_features = self._run_inner_cv(X_train, y_train)
            
            # Prediction
            y_pred = best_pipeline.predict(X_test)
            y_proba = best_pipeline.predict_proba(X_test) if hasattr(best_pipeline.named_steps['model'], 'predict_proba') else None
            
            test_metrics = calculate_metrics(y_test, y_pred, y_proba)
            print(f"  Test - Accuracy: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}")
            
            fold_result = FoldResults(
                fold_index=outer_fold,
                y_true=y_test,
                y_pred=y_pred,
                y_proba=y_proba,
                metrics=test_metrics,
                selected_features=selected_features
            )
            results.add_fold_result(fold_result)
        
        results.calculate_overall_metrics()
        results.overall_metrics["total_runtime_seconds"] = time.time() - start_time_round
        
        # Save results
        if round_idx is None:
            save_path = self.run_dir / "results.json"
        else:
            save_path = self.run_dir / f"round{round_idx + 1}.json"
        
        results.save_json(save_path)
        return results

    def run(self) -> Union[ExperimentResults, List[ExperimentResults]]:
        """Main entry point to execute the experiment."""
        if isinstance(self.data_fold_idxs, list):
            print(f"Starting Multi-Round Experiment: {self.name}")
            all_results = []
            n_rounds = len(self.data_fold_idxs)
            for i in range(n_rounds):
                print(f"\n### Round {i + 1}/{n_rounds} ###")
                all_results.append(self.run_single_round(round_idx=i))
            return all_results
        else:
            print(f"Starting Single-Round Experiment: {self.name}")
            return self.run_single_round()
