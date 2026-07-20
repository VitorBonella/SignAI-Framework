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
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

from sklearn.metrics import balanced_accuracy_score, f1_score as sk_f1_score

from signalai.core.experiment import BaseExperiment
from signalai.utils.metrics import calculate_metrics
from signalai.utils.results import ExperimentResults, FoldResults

class TorchVibrationDataset(Dataset):
    """Wrapper to convert dataset samples into Torch tensors."""
    def __init__(self, X: np.ndarray, y: np.ndarray, is_reconstruction: bool = False):
        self.X = torch.tensor(X, dtype=torch.float32)
        if is_reconstruction:
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = torch.tensor(y, dtype=torch.long)
        self.is_reconstruction = is_reconstruction

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
        data_fold_idxs: Union[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]],
        model: nn.Module,
        criterion: Optional[nn.Module] = None,
        optimizer_class: Optional[torch.optim.Optimizer] = optim.Adam,
        batch_size: int = 32,
        lr: float = 1e-3,
        num_epochs: int = 20,
        val_split: float = 0.2,
        output_dir: str = "results_torch",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        is_reconstruction: bool = False,
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
        self.is_reconstruction = is_reconstruction
        
        if criterion is not None:
            self.criterion = criterion
        else:
            self.criterion = nn.MSELoss() if is_reconstruction else nn.CrossEntropyLoss()

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = torch.nn.DataParallel(self.model)
        else:
            self.model = self.model.to(self.device)

        if isinstance(data_fold_idxs, list):
            self.n_outer_folds = len(data_fold_idxs)
        else:
            self.n_outer_folds = len(np.unique(data_fold_idxs))
        
        self.prepare_data()

    def prepare_data(self):
        features, labels = [], []
        for i, sample in enumerate(self.dataset):
            sig = sample['signal'][0]
            if isinstance(sig, list):
                sig = np.array(sig)
            
            if sig.ndim == 1:
                sig = sig[np.newaxis, :]
            
            features.append(sig)
            labels.append(sample['metainfo']['label'])

        self.X = np.array(features)
        
        if not self.is_reconstruction:
            self.le = LabelEncoder()
            self.y = self.le.fit_transform(labels)
            self.n_classes = len(self.le.classes_)
        else:
            self.y = self.X # y is same as X for reconstruction
            self.n_classes = None

    def plot_losses(self, train_losses: List[float], val_losses: List[float], fold_val: int, prefix: str = ""):
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        if val_losses:
            plt.plot(val_losses, label='Validation Loss')
        plt.title(f'Training and Validation Loss - Fold {fold_val}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plot_path = self.run_dir / f"{prefix}loss_fold_{fold_val}.png"
        plt.savefig(plot_path)
        plt.close()

    def _train_one_fold(self, X_train, y_train, X_test, y_test, fold_val: int, prefix: str = "") -> FoldResults:
        train_dataset = TorchVibrationDataset(X_train, y_train, is_reconstruction=self.is_reconstruction)
        test_dataset = TorchVibrationDataset(X_test, y_test, is_reconstruction=self.is_reconstruction)

        val_size = int(self.val_split * len(train_dataset))
        train_size = len(train_dataset) - val_size
        if val_size > 0:
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            val_loader = None

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        model = copy.deepcopy(self.model).to(self.device)
        optimizer = self.optimizer_class(model.parameters(), lr=self.lr)

        has_conv = any(isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)) for m in model.modules())

        fold_model_path = self.run_dir / f"{prefix}model_fold_{fold_val}.pth"
        best_val_f1 = -1.0
        best_val_loss = float('inf')

        print(f"\n  {'Epoch':>6} | {'Time':>6} | {'Train Loss':>10} | {'Val Loss':>9} | {'Val Bal.Acc':>11} | {'Val F1':>8} | {'Saved':>5}")
        print(f"  {'-'*6}-+-{'-'*6}-+-{'-'*10}-+-{'-'*9}-+-{'-'*11}-+-{'-'*8}-+-{'-'*5}")

        train_losses, val_losses = [], []
        for epoch in range(self.num_epochs):
            epoch_start = time.time()

            model.train()
            running_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                if xb.ndim == 2 and has_conv:
                    xb = xb.unsqueeze(1)
                optimizer.zero_grad()
                outputs = model(xb)
                loss = self.criterion(outputs, yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * xb.size(0)

            avg_train_loss = running_loss / len(train_loader.dataset)
            train_losses.append(avg_train_loss)

            epoch_time = time.time() - epoch_start
            saved = False

            if val_loader:
                model.eval()
                val_loss = 0.0
                val_true, val_pred = [], []
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(self.device), yb.to(self.device)
                        if xb.ndim == 2 and has_conv:
                            xb = xb.unsqueeze(1)
                        outputs = model(xb)
                        loss = self.criterion(outputs, yb)
                        val_loss += loss.item() * xb.size(0)
                        if not self.is_reconstruction:
                            preds = torch.argmax(outputs, dim=1)
                            val_true.extend(yb.cpu().numpy())
                            val_pred.extend(preds.cpu().numpy())
                avg_val_loss = val_loss / len(val_loader.dataset)
                val_losses.append(avg_val_loss)

                if not self.is_reconstruction and val_true:
                    val_ba = balanced_accuracy_score(val_true, val_pred)
                    val_f1 = sk_f1_score(val_true, val_pred, average='macro', zero_division=0)
                    if val_f1 > best_val_f1:
                        best_val_f1 = val_f1
                        torch.save(model.state_dict(), fold_model_path)
                        saved = True
                    print(f"  {epoch+1:>6}/{self.num_epochs} | {epoch_time:>5.1f}s | {avg_train_loss:>10.4f} | {avg_val_loss:>9.4f} | {val_ba:>11.4f} | {val_f1:>8.4f} | {'*':>5}" if saved else
                          f"  {epoch+1:>6}/{self.num_epochs} | {epoch_time:>5.1f}s | {avg_train_loss:>10.4f} | {avg_val_loss:>9.4f} | {val_ba:>11.4f} | {val_f1:>8.4f} | {'':>5}")
                else:
                    # Reconstruction: save on best val loss
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        torch.save(model.state_dict(), fold_model_path)
                        saved = True
                    print(f"  {epoch+1:>6}/{self.num_epochs} | {epoch_time:>5.1f}s | {avg_train_loss:>10.4f} | {avg_val_loss:>9.4f} | {'N/A':>11} | {'N/A':>8} | {'*':>5}" if saved else
                          f"  {epoch+1:>6}/{self.num_epochs} | {epoch_time:>5.1f}s | {avg_train_loss:>10.4f} | {avg_val_loss:>9.4f} | {'N/A':>11} | {'N/A':>8} | {'':>5}")
            else:
                # No validation: save last epoch
                torch.save(model.state_dict(), fold_model_path)
                saved = True
                print(f"  {epoch+1:>6}/{self.num_epochs} | {epoch_time:>5.1f}s | {avg_train_loss:>10.4f} | {'N/A':>9} | {'N/A':>11} | {'N/A':>8} | {'*':>5}" if saved else
                      f"  {epoch+1:>6}/{self.num_epochs} | {epoch_time:>5.1f}s | {avg_train_loss:>10.4f} | {'N/A':>9} | {'N/A':>11} | {'N/A':>8} | {'':>5}")

        # Plot losses
        self.plot_losses(train_losses, val_losses, fold_val, prefix=prefix)

        # Load best checkpoint for final evaluation
        model.load_state_dict(torch.load(fold_model_path, map_location=self.device))

        # Evaluation
        y_true, y_pred, y_proba = [], [], []
        if not self.is_reconstruction:
            model.eval()
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    if xb.ndim == 2 and has_conv:
                        xb = xb.unsqueeze(1)
                    outputs = model(xb)
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    y_true.extend(yb.cpu().numpy())
                    y_pred.extend(preds.cpu().numpy())
                    y_proba.extend(probs.cpu().numpy())
            
            metrics = calculate_metrics(np.array(y_true), np.array(y_pred), np.array(y_proba))
        else:
            # For reconstruction, we might want to save some metrics or just return dummy
            metrics = {"mse": train_losses[-1]}
            y_true, y_pred, y_proba = [0], [0], [[0]]

        return FoldResults(
            fold_index=fold_val,
            y_true=np.array(y_true),
            y_pred=np.array(y_pred),
            y_proba=np.array(y_proba),
            metrics=metrics
        )

    def run_single_round(self, round_idx: int = None) -> ExperimentResults:
        """Runs training and evaluation for a single round of cross-validation."""
        assignments = self.data_fold_idxs[round_idx] if round_idx is not None else self.data_fold_idxs
        prefix = f"round{round_idx + 1}_" if round_idx is not None else ""

        results = ExperimentResults(
            experiment_name=self.name,
            description=self.description,
            model_name=self.model.__class__.__name__,
            feature_names=None,
            config={
                'epochs': self.num_epochs,
                'batch_size': self.batch_size,
                'learning_rate': self.lr,
                'val_split': self.val_split,
                'round': round_idx
            }
        )

        unique_folds = np.unique(assignments)
        for fold_val in unique_folds:
            print(f"  Fold {fold_val}/{len(unique_folds) - 1}...")
            train_mask = assignments != fold_val
            test_mask = assignments == fold_val

            X_train, X_test = self.X[train_mask], self.X[test_mask]
            y_train, y_test = self.y[train_mask], self.y[test_mask]

            if len(X_train) == 0 or len(X_test) == 0:
                print(f"  Skipping empty fold {fold_val}")
                continue

            fold_result = self._train_one_fold(X_train, y_train, X_test, y_test, fold_val=fold_val, prefix=prefix)
            results.add_fold_result(fold_result)

        results.calculate_overall_metrics()

        if round_idx is None:
            save_path = self.run_dir / "results.json"
        else:
            save_path = self.run_dir / f"round{round_idx + 1}.json"

        results.save_json(save_path)
        return results

    def run(self) -> Union[ExperimentResults, List[ExperimentResults]]:
        """Main entry point. Mirrors ClassificationExperiment: single round → results.json,
        multi-round → round1.json, round2.json, ..."""
        if isinstance(self.data_fold_idxs, list):
            print(f"Starting Multi-Round DL Experiment: {self.name}")
            all_results = []
            n_rounds = len(self.data_fold_idxs)
            for i in range(n_rounds):
                print(f"\n### Round {i + 1}/{n_rounds} ###")
                all_results.append(self.run_single_round(round_idx=i))
            return all_results
        else:
            print(f"Starting Single-Round DL Experiment: {self.name}")
            return self.run_single_round()
