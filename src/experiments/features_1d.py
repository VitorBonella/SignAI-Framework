# vibclassifier/experiments/features_1d.py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, List, Callable, Dict, Optional, Tuple, Union
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from vibdata.deep.signal.transforms import Transform
from vibdata.deep.DeepDataset import DeepDataset
from .base import Experiment
from src.utils.metrics import calculate_metrics
from src.utils.experiment_result import ExperimentResults, FoldResults

import time
import os

class Features1DExperiment(Experiment):

    
    def __init__(
        self,
        name: str,
        description: str,
        dataset: DeepDataset = None,
        data_fold_idxs: List[int] | List[List[int]] = None,
        feature_names: List[str] = None,
        n_inner_folds: int = 3,
        feature_selector: Optional[Callable] = None,
        output_dir: str = "results_1d_features",
        random_state: int = 42,
        model_parameters_search_space: Optional[Dict[str, Any]] = None,
        scaler: Optional[Callable] = StandardScaler,
        **kwargs
    ):
        super().__init__(name, description, dataset, **kwargs)
        self.n_inner_folds = n_inner_folds

        if not isinstance(feature_names[0], str):
            feature_names = [f.__class__.__name__ for f in feature_names]

        self.feature_names = feature_names
        self.feature_selector = feature_selector
        self.output_dir = Path(output_dir)
        self.random_state = random_state
        self.data_fold_idxs = data_fold_idxs
        self.results = None
        self.model_parameters_search_space = model_parameters_search_space
        self.scaler = scaler if scaler is not None else StandardScaler

        if isinstance(data_fold_idxs[0], list):
            self.n_outer_folds = len(np.unique(data_fold_idxs[0]))
        else:
            self.n_outer_folds = len(np.unique(data_fold_idxs))
        

        self.prepare_data()

        # Criar diretório se não existir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def prepare_data(self):
        #prepara x e y
        features, labels = [], []
        for sample in self.dataset:
            features.append(sample['signal'][0])
            labels.append(sample['metainfo']['label'])
        
        # Criar arrays numpy
        self.X = np.array(features)
        self.y = np.array(labels)

    def _create_pipeline(self) -> Pipeline:
        """Cria o pipeline de processamento com seleção de features opcional."""
        steps = [
            ('scaler', self.scaler())
        ]
        
        if self.feature_selector is not None:
            steps.append(('feature_selector', self.feature_selector))
        
        if self.model is None:
            raise ValueError("Modelo não foi definido para o experimento")
            
        steps.append(('model', self.model))
        return Pipeline(steps)

    def _run_inner_cv(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Tuple[Dict, Pipeline, Optional[List[str]]]:
        """
        Executa validação cruzada interna para ajuste de modelo.
        
        Args:
            X_train: Dados de treino
            y_train: Rótulos de treino
            
        Returns:
            Tuple contendo:
                - Métricas médias da validação interna
                - Melhor pipeline encontrado
                - Features selecionadas (se aplicável)
        """
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
        
        # Fit the search
        search.fit(X_train, y_train)
        
        # Get best pipeline and results
        best_pipeline = search.best_estimator_
        cv_results = search.cv_results_

        # Store inner fold results
        inner_results = []
        for i in range(self.n_inner_folds):
            fold_metrics = {
                'f1': cv_results[f'split{i}_test_score'][search.best_index_],
            }
            inner_results.append(fold_metrics)
        
    
        avg_metrics = {
            'mean_f1': np.mean([m['f1'] for m in inner_results]),
            'std_f1': np.std([m['f1'] for m in inner_results]),
            'inner_folds_details': inner_results,
            'best_params': search.best_params_
        }
        
        # Get selected features if applicable
        selected_features = None
        if 'feature_selector' in best_pipeline.named_steps:
            selector = best_pipeline.named_steps['feature_selector']
            selected_features = [
                self.feature_names[i] for i in selector.get_support(indices=True)
            ]

        return avg_metrics, best_pipeline, selected_features
    
    def run_single_round(self, multi_round_i = None) -> ExperimentResults:
        """Executa o experimento e retorna objeto ExperimentResults."""
        if self.model is None:
            raise ValueError("Modelo não foi definido para o experimento")
        
        X = self.X
        y = self.y

        if multi_round_i is None:
            folds = self.data_fold_idxs
        else:
            if not isinstance(self.data_fold_idxs[0], np.ndarray):
                raise ValueError("data_fold_idxs deve ser uma lista de listas para múltiplas rodadas.")
            if multi_round_i < 0 or multi_round_i >= len(self.data_fold_idxs):
                raise ValueError(f"Índice i={multi_round_i} fora do intervalo para data_fold_idxs com {len(self.data_fold_idxs)} rodadas.")
            folds = self.data_fold_idxs[multi_round_i]

        # Criar objeto para armazenar resultados
        results = ExperimentResults(
            experiment_name=self.name,
            description=self.description,
            model_name=self.model.__class__.__name__,
            feature_names=self.feature_names,
            config={
                'n_outer_folds': self.n_outer_folds,
                'n_inner_folds': self.n_inner_folds,
                #'feature_selector': self.feature_selector.__name__,
                'random_state': self.random_state,
                'data_transform': str(self.data_transform.__class__.__name__) 
                    if self.data_transform else None
            }
        )
        
        # Validação cruzada aninhada
        for outer_fold in range(self.n_outer_folds):
            print(f"\n=== Fold Externo {outer_fold + 1}/{self.n_outer_folds} ===")
            
            # Divisão dos dados
            train_mask = folds != outer_fold
            test_mask = folds == outer_fold
            
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
            
            # Validação cruzada interna
            inner_metrics, best_pipeline, selected_features = self._run_inner_cv(X_train, y_train)
            
            # Treinar com todos os dados de treino
            best_pipeline.fit(X_train, y_train)
            
            # Prever no conjunto de teste
            y_pred = best_pipeline.predict(X_test)
            y_proba = None
            if hasattr(best_pipeline.named_steps['model'], 'predict_proba'):
                y_proba = best_pipeline.predict_proba(X_test)
            
            # Calcular métricas
            test_metrics = calculate_metrics(y_test, y_pred, y_proba)
            
            # Armazenar importância das features se disponível
            #feature_importances = None
            #if hasattr(best_pipeline.named_steps['model'], 'feature_importances_'):
            #    feature_importances = dict(zip(
            #        self.feature_names,
            #        best_pipeline.named_steps['model'].feature_importances_
            #    ))
            
            # Criar resultado do fold
            fold_result = FoldResults(
                fold_index=outer_fold,
                y_true=y_test,
                y_pred=y_pred,
                y_proba=y_proba,
                metrics=test_metrics,
                selected_features=selected_features,
                #feature_importances=feature_importances
            )
            #print("FOLD RESULT>",fold_result)
            results.add_fold_result(fold_result)
            
            print(f"  Teste - Acurácia: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}")
        
        # Calcular métricas agregadas
        results.calculate_overall_metrics()
        
        print("\n=== Resultados Finais ===")
        print(f"Acurácia Média: {results.overall_metrics['accuracy']:.4f} ± {results.overall_metrics['std_accuracy']:.4f}")
        print(f"F1-Score Médio: {results.overall_metrics['mean_f1']:.4f} ± {results.overall_metrics['std_f1']:.4f}")
        
        if multi_round_i is None:
            results.save_json(f"vibration_analysis_results_{results.experiment_name}_{self.start_time}.json")
        else:
            #create a directory if not exists
            dir_path = f"{self.output_dir}/vibration_analysis_results_{results.experiment_name}_{self.start_time}"
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                print(f"Changed working directory to: {dir_path}")
            # Save results
            results.save_json(f"{dir_path}/round{multi_round_i+1}.json")

        return results
    
    def run_multi_round(self) -> List[ExperimentResults]:
        is_valid_type = isinstance(self.data_fold_idxs[0], np.ndarray)
        if self.data_fold_idxs is None or not is_valid_type:
            raise ValueError("data_fold_idxs deve ser uma lista de listas para múltiplas rodadas.")
        
        all_results = []
        for round_idx, fold_idxs in enumerate(self.data_fold_idxs):
            print(f"\n### Rodada {round_idx + 1}/{len(self.data_fold_idxs)} ###")
            all_results.append(self.run_single_round(multi_round_i=round_idx))
        
        return all_results
    
    def run(self) -> ExperimentResults:
        """Executa o experimento e retorna objeto ExperimentResults se for uma única rodada. Se nao for, use run_multi_round()."""
        self.start_time = time.strftime("%Y%m%d_%H%M%S")
        if isinstance(self.data_fold_idxs[0], np.ndarray):
            return self.run_multi_round()
        else:
            return self.run_single_round()
        