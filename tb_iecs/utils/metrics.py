"""
Metrics calculation utilities
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix,
    cohen_kappa_score, precision_recall_curve, auc
)
from typing import Dict, List, Tuple, Union, Optional, Any


def calculate_enrichment_factor(y_true: np.ndarray, y_score: np.ndarray, top: float = 0.01) -> float:
    """
    Calculate enrichment factor at a specific percentage cutoff.
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_score: Predicted probabilities or scores
        top: Top fraction of compounds to consider (0-1)
        
    Returns:
        Enrichment factor value
    """
    # Create dataframe with true labels and scores
    df = pd.DataFrame({"y_true": y_true, "y_score": y_score})
    
    # Sort by prediction score (descending)
    df = df.sort_values(by="y_score", ascending=False)
    
    # Calculate total counts
    N_total = len(df)
    N_active = len(df[df["y_true"] == 1])
    
    # Calculate counts in top fraction
    top_n = int(N_total * top + 0.5)  # Round to nearest integer
    df_top = df.iloc[:top_n]
    top_active = len(df_top[df_top["y_true"] == 1])
    
    # Calculate enrichment factor
    if N_active == 0 or top_n == 0:  # Avoid division by zero
        return 0.0
    
    random_ratio = N_active / N_total
    observed_ratio = top_active / top_n
    
    return observed_ratio / random_ratio


def calculate_bedroc(y_true: np.ndarray, y_score: np.ndarray, alpha: float = 20.0) -> float:
    """
    Calculate Boltzmann-Enhanced Discrimination of ROC (BEDROC).
    BEDROC addresses the "early recognition" problem in virtual screening.
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_score: Predicted probabilities or scores
        alpha: Parameter controlling the magnitude of early recognition
        
    Returns:
        BEDROC score (between 0 and 1)
    """
    if len(y_true) != len(y_score):
        raise ValueError("Input arrays must have the same length")
    
    n_actives = sum(y_true)
    if n_actives == 0:
        return 0.0
    
    # Create dataframe and sort by score
    df = pd.DataFrame({"y_true": y_true, "y_score": y_score})
    df = df.sort_values(by="y_score", ascending=False)
    
    # Get ranks of actives
    ranks = [i+1 for i, (_, row) in enumerate(df.iterrows()) if row["y_true"] == 1]
    
    # Calculate sum of exponentials
    sum_exp = sum(np.exp(-alpha * rank / len(y_true)) for rank in ranks)
    
    # Calculate BEDROC
    ri = n_actives / len(y_true)
    bedroc = (sum_exp / n_actives) * ((np.exp(alpha/len(y_true)) - 1) / 
                                       (np.exp(alpha) - np.exp(alpha * ri)))
    
    # Normalize to [0, 1]
    random = ri * ((np.exp(alpha/len(y_true)) - 1) / 
                  (np.exp(alpha) - np.exp(alpha * ri)))
    return (bedroc - random) / (1 - random)


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """
    Calculate a comprehensive set of metrics for binary classification.
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        y_score: Predicted probabilities or scores
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic classification metrics
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)
    metrics["kappa"] = cohen_kappa_score(y_true, y_pred)
    
    # ROC and PR metrics
    if len(np.unique(y_true)) > 1:  # Check if both classes present
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
        
        # Calculate PR AUC
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        metrics["pr_auc"] = auc(recall, precision)
    else:
        metrics["roc_auc"] = 0.5
        metrics["pr_auc"] = 0.5
    
    # Confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics["tn"] = int(tn)
    metrics["fp"] = int(fp)
    metrics["fn"] = int(fn)
    metrics["tp"] = int(tp)
    
    # Virtual screening specific metrics
    metrics["ef_1"] = calculate_enrichment_factor(y_true, y_score, top=0.01)
    metrics["ef_2"] = calculate_enrichment_factor(y_true, y_score, top=0.02)
    metrics["ef_5"] = calculate_enrichment_factor(y_true, y_score, top=0.05)
    metrics["ef_10"] = calculate_enrichment_factor(y_true, y_score, top=0.10)
    metrics["bedroc"] = calculate_bedroc(y_true, y_score)
    
    return metrics 