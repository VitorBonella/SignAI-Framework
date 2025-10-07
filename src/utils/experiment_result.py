# vibclassifier/utils/experiment_results.py
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, asdict, field
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class FoldResults:
    """Armazena resultados de um fold específico."""
    fold_index: int
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: Optional[np.ndarray] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    selected_features: Optional[List[str]] = None
    feature_importances: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        # Garantir que a matriz de confusão seja calculada
        if len(self.confusion_matrix) == 0 and len(self.y_true) > 0:
            self.confusion_matrix = confusion_matrix(self.y_true, self.y_pred)
    
    def to_dict(self) -> Dict:
        """Converte para dicionário serializável."""
        data = asdict(self)
        # Converter numpy arrays para listas
        for key in ['y_true', 'y_pred', 'confusion_matrix']:
            if key in data and isinstance(data[key], np.ndarray):
                data[key] = data[key].tolist()
        
        if 'y_proba' in data and data['y_proba'] is not None:
            data['y_proba'] = data['y_proba'].tolist()
        
        return data

@dataclass
class ExperimentResults:
    """Armazena todos os resultados de um experimento."""
    experiment_name: str
    description: str
    model_name: str
    feature_names: List[str]
    folds: List[FoldResults] = field(default_factory=list)
    overall_metrics: Dict[str, float] = field(default_factory=dict)
    config: Dict = field(default_factory=dict)
    
    def add_fold_result(self, fold_result: FoldResults):
        """Adiciona resultados de um fold."""
        self.folds.append(fold_result)
    
    def calculate_overall_metrics(self):
        """Calcula métricas agregadas de todos os folds."""
        all_y_true = np.concatenate([fold.y_true for fold in self.folds])
        all_y_pred = np.concatenate([fold.y_pred for fold in self.folds])
        
        # Matriz de confusão geral
        overall_cm = confusion_matrix(all_y_true, all_y_pred)
        
        # Calcular métricas globais
        self.overall_metrics = {
            'accuracy': np.mean([fold.metrics['accuracy'] for fold in self.folds]),
            'std_accuracy': np.std([fold.metrics['accuracy'] for fold in self.folds]),
            'mean_f1': np.mean([fold.metrics['f1'] for fold in self.folds]),
            'std_f1': np.std([fold.metrics['f1'] for fold in self.folds]),
            'confusion_matrix': overall_cm.tolist()
        }
    
    def to_dict(self) -> Dict:
        """Converte para dicionário serializável."""
        data = asdict(self)
        data['folds'] = [fold.to_dict() for fold in self.folds]
        return data
    
    def save_json(self, filepath: Union[str, Path]):
        """Salva resultados em arquivo JSON."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
    
    @classmethod
    def load_json(cls, filepath: Union[str, Path]) -> 'ExperimentResults':
        """Carrega resultados de arquivo JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruir objetos FoldResults
        folds = []
        for fold_data in data['folds']:
            fold = FoldResults(
                fold_index=fold_data['fold_index'],
                y_true=np.array(fold_data['y_true']),
                y_pred=np.array(fold_data['y_pred']),
                y_proba=np.array(fold_data['y_proba']) if fold_data['y_proba'] else None,
                metrics=fold_data['metrics'],
                confusion_matrix=np.array(fold_data['confusion_matrix']),
                selected_features=fold_data['selected_features'],
                feature_importances=fold_data['feature_importances']
            )
            folds.append(fold)
        
        # Reconstruir ExperimentResults
        return cls(
            experiment_name=data['experiment_name'],
            description=data['description'],
            model_name=data['model_name'],
            feature_names=data['feature_names'],
            folds=folds,
            overall_metrics=data['overall_metrics'],
            config=data['config']
        )
    
    def show_results(self, figsize: tuple = (10, 6), cmap: str = 'Blues'):
        """
        Exibe os resultados do experimento em formato de tabela e matriz de confusão.
        
        Args:
            figsize: Tamanho da figura da matriz de confusão
            cmap: Mapa de cores para a matriz de confusão
        """
        # Verificar se as métricas gerais foram calculadas
        if not self.overall_metrics:
            self.calculate_overall_metrics()
        
        print("=" * 70)
        print(f"EXPERIMENTO: {self.experiment_name}")
        print(f"DESCRIÇÃO: {self.description}")
        print(f"MODELO: {self.model_name}")
        print(f"QUANTIDADE DE FOLDS: {len(self.folds)}")
        if self.feature_names:
            print(f"QUANTIDADE DE FEATURES: {len(self.feature_names)}")
        print("=" * 70)
        
        # Criar tabela de métricas gerais
        metrics_data = {
            'Métrica': [
                'Acurácia Média',
                'Desvio Padrão Acurácia', 
                'F1-Score Médio',
                'Desvio Padrão F1-Score'
            ],
            'Valor': [
                f"{self.overall_metrics['accuracy']:.4f}",
                f"±{self.overall_metrics['std_accuracy']:.4f}",
                f"{self.overall_metrics['mean_f1']:.4f}",
                f"±{self.overall_metrics['std_f1']:.4f}"
            ]
        }
        
        df_metrics = pd.DataFrame(metrics_data)
        print("\nMÉTRICAS GERAIS DO EXPERIMENTO:")
        print("-" * 50)
        print(df_metrics.to_string(index=False))
        
        # Tabela detalhada por fold
        if self.folds:
            print(f"\nDETALHES POR FOLD:")
            print("-" * 60)
            fold_data = []
            for i, fold in enumerate(self.folds):
                fold_data.append({
                    'Fold': i + 1,
                    'Acurácia': f"{fold.metrics.get('accuracy', 0):.4f}",
                    'F1-Score': f"{fold.metrics.get('f1', 0):.4f}",
                    'Precision': f"{fold.metrics.get('precision', 0):.4f}",
                    'Recall': f"{fold.metrics.get('recall', 0):.4f}"
                })
            
            df_folds = pd.DataFrame(fold_data)
            print(df_folds.to_string(index=False))
        
        # Plotar matriz de confusão
        if 'confusion_matrix' in self.overall_metrics:
            self._plot_confusion_matrix(
                np.array(self.overall_metrics['confusion_matrix']),
                figsize=figsize,
                cmap=cmap
            )
    
    def _plot_confusion_matrix(self, cm: np.ndarray, figsize: tuple = (10, 6), cmap: str = 'Blues'):
        """
        Plota a matriz de confusão.
        
        Args:
            cm: Matriz de confusão
            figsize: Tamanho da figura
            cmap: Mapa de cores
        """
        plt.figure(figsize=figsize)
        
        # Calcular totais para anotações
        cm_sum = np.sum(cm, axis=1, keepdims=True)
        cm_percentage = cm / cm_sum.astype(float) * 100
        
        # Criar heatmap
        ax = sns.heatmap(
            cm, 
            annot=True, 
            fmt='d',
            cmap=cmap,
            cbar=True,
            square=True,
            linewidths=0.5,
            linecolor='white'
        )
        
        # Adicionar porcentagens
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if cm[i, j] > 0:  # Só adiciona texto se o valor for maior que 0
                    ax.text(j + 0.5, i + 0.3, f'{cm_percentage[i, j]:.1f}%', 
                           ha='center', va='center', fontsize=10, color='red')
        
        plt.title(f'Confusion Matrix - {self.experiment_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('True', fontsize=12)
        
        # Ajustar layout
        plt.tight_layout()
        plt.show()
        
        # Estatísticas adicionais da matriz de confusão
        accuracy = np.trace(cm) / np.sum(cm)
        print(f"\nESTATÍSTICAS DA MATRIZ DE CONFUSÃO:")
        print(f"Acurácia Geral: {accuracy:.4f}")
        print(f"Total de Amostras: {np.sum(cm)}")