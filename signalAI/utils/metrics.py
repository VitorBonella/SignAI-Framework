# vibclassifier/utils/metrics.py
import numpy as np
from typing import List, Optional, Dict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    labels: Optional[List] = None
) -> Dict:
    """
    Calcula métricas completas de classificação.
    
    Args:
        y_true: Valores reais
        y_pred: Valores preditos
        y_proba: Probabilidades preditas (opcional)
        labels: Nomes das classes (opcional)
        
    Returns:
        Dicionário com todas as métricas calculadas
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'classification_report': classification_report(y_true, y_pred, output_dict=True)
    }
    
    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            metrics['roc_auc'] = roc_auc_score(
                y_true, y_proba, multi_class='ovr', average='weighted')
        except Exception as e:
            metrics['roc_auc'] = None
            metrics['roc_auc_error'] = str(e)
    
    return metrics