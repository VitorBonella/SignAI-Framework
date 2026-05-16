import os
import time
import copy
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

from signalai.core.experiment import BaseExperiment
from signalai.utils.metrics import calculate_metrics
from signalai.utils.results import ExperimentResults, FoldResults

class TorchVibrationDataset(Dataset):
    """Wrapper to convert dataset samples into Torch tensors."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class DeepLearningExperiment(BaseExperiment):
    """Experiment for deep learning models using PyTorch."""
    
    def __init__(
        self,
        name: str,
        description: str,
        dataset,
        data_fold_idxs: np.ndarray,
        model: nn.Module,
        criterion: Optional[nn.Module] = None,
        optimizer_class: Optional[torch.optim.Optimizer] = optim.Adam,
        batch_size: int = 32,
        lr: float = 1e-3,
        num_epochs: int = 20,
        val_split: float = 0.2,
        output_dir: str = "results_torch",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ):
        super().__init__(name, description, dataset, model=model, output_dir=output_dir, **kwargs)
        self.data_fold_idxs = data_fold_idxs
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.val_split = val_split
        self.device = device
        self.optimizer_class = optimizer_class
        self.lr = lr
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = torch.nn.DataParallel(self.model)

        self.n_outer_folds = len(np.unique(data_fold_idxs))
        self.prepare_data()

    def prepare_data(self):
        features, labels = [], []
        for sample in self.dataset:
            features.append(sample['signal'][0])
            labels.append(sample['metainfo']['label'])

        self.X = np.array(features)
        self.y = np.array(labels)

    def _train_one_fold(self, X_train, y_train, X_test, y_test, fold_idx: int) -> FoldResults:
        train_dataset = TorchVibrationDataset(X_train, y_train)
        test_dataset = TorchVibrationDataset(X_test, y_test)

        val_size = int(self.val_split * len(train_dataset))
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        model = copy.deepcopy(self.model).to(self.device)
        optimizer = self.optimizer_class(model.parameters(), lr=self.lr)

        train_losses, val_losses = [], []
        for epoch in range(self.num_epochs):
            model.train()
            running_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                
                # Reshape for Conv layers if necessary
                if xb.ndim == 2:
                    if any(isinstance(m, nn.Conv1d) for m in model.modules()):
                        xb = xb.unsqueeze(1)
                    elif any(isinstance(m, (nn.Conv2d, nn.DataParallel)) for m in model.modules()):
                        side = int(np.sqrt(xb.shape[1]))
                        if side * side == xb.shape[1]:
                            xb = xb.view(xb.size(0), 1, side, side)

                optimizer.zero_grad()
                outputs = model(xb)
                loss = self.criterion(outputs, yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * xb.size(0)

            avg_train_loss = running_loss / len(train_loader.dataset)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    if xb.ndim == 2:
                        if any(isinstance(m, nn.Conv1d) for m in model.modules()):
                            xb = xb.unsqueeze(1)
                        elif any(isinstance(m, (nn.Conv2d, nn.DataParallel)) for m in model.modules()):
                            side = int(np.sqrt(xb.shape[1]))
                            if side * side == xb.shape[1]:
                                xb = xb.view(xb.size(0), 1, side, side)
                    outputs = model(xb)
                    loss = self.criterion(outputs, yb)
                    val_loss += loss.item() * xb.size(0)

            avg_val_loss = val_loss / len(val_loader.dataset)
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

        # Evaluation
        y_true, y_pred, y_proba = [], [], []
        model.eval()
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                if xb.ndim == 2:
                    if any(isinstance(m, nn.Conv1d) for m in model.modules()):
                        xb = xb.unsqueeze(1)
                    elif any(isinstance(m, (nn.Conv2d, nn.DataParallel)) for m in model.modules()):
                        side = int(np.sqrt(xb.shape[1]))
                        if side * side == xb.shape[1]:
                            xb = xb.view(xb.size(0), 1, side, side)
                outputs = model(xb)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                y_true.extend(yb.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                y_proba.extend(probs.cpu().numpy())

        return FoldResults(
            fold_index=fold_idx,
            y_true=np.array(y_true),
            y_pred=np.array(y_pred),
            y_proba=np.array(y_proba),
            metrics=calculate_metrics(np.array(y_true), np.array(y_pred), np.array(y_proba))
        )

    def run(self) -> ExperimentResults:
        results = ExperimentResults(
            experiment_name=self.name,
            description=self.description,
            model_name=self.model.__class__.__name__,
            feature_names=None,
            config={
                'n_outer_folds': self.n_outer_folds,
                'epochs': self.num_epochs,
                'batch_size': self.batch_size,
                'learning_rate': self.lr,
                'val_split': self.val_split
            }
        )

        for outer_fold in range(self.n_outer_folds):
            print(f"Outer Fold {outer_fold+1}/{self.n_outer_folds}...")
            train_mask = self.data_fold_idxs != outer_fold
            test_mask = self.data_fold_idxs == outer_fold
            X_train, X_test = self.X[train_mask], self.X[test_mask]
            y_train, y_test = self.y[train_mask], self.y[test_mask]

            fold_result = self._train_one_fold(X_train, y_train, X_test, y_test, fold_idx=outer_fold)
            results.add_fold_result(fold_result)

        results.calculate_overall_metrics()
        results.save_json(self.run_dir / "torch_results.json")
        return results
