import numpy as np
from typing import List, Optional, Dict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, balanced_accuracy_score
)

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    labels: Optional[List] = None
) -> Dict:
    """
    Calculates comprehensive classification metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        y_proba: Predicted probabilities (optional)
        labels: Class names (optional)
        
    Returns:
        Dictionary with all calculated metrics
    """
    metrics = {
        'accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro'),
        'recall': recall_score(y_true, y_pred, average='macro'),
        'f1': f1_score(y_true, y_pred, average='macro'),
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
