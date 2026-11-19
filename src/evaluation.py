"""
Comprehensive model evaluation metrics and plotting utilities for AML detection.
Calculates:
- Precision, Recall, F1, F2-score
- Average Precision (PR-AUC)
- ROC-AUC
- Confusion matrix (TP, FP, TN, FN)
- False Positive Rate (FPR), False Negative Rate (FNR)
- Optimal threshold search based on validation PR curve / F2-score / F1-score
"""

import os
import json
import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score, fbeta_score,
    average_precision_score, roc_auc_score,
    confusion_matrix, precision_recall_curve, roc_curve
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f2",
    beta: float = 2.0
) -> Tuple[float, float]:
    """
    Find optimal classification threshold on validation data.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_prob : np.ndarray
        Predicted probabilities for positive class.
    metric : str
        Optimization goal ('f2', 'f1', or 'recall_precision_balanced').
    beta : float
        Beta value for fbeta_score (default 2.0 to prioritize recall).
        
    Returns
    -------
    Tuple[float, float]
        (best_threshold, best_metric_score)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # thresholds has length len(precisions)-1
    
    best_score = -1.0
    best_thresh = 0.5

    for p, r, thresh in zip(precisions[:-1], recalls[:-1], thresholds):
        if p + r == 0:
            continue
        if metric == "f2":
            score = (1 + beta**2) * (p * r) / ((beta**2 * p) + r)
        elif metric == "f1":
            score = 2 * (p * r) / (p + r)
        elif metric == "balanced":
            # harmonic balance prioritizing high recall >= 0.70 with precision
            score = (p * r) if r >= 0.70 else 0.0
        else:
            score = (1 + beta**2) * (p * r) / ((beta**2 * p) + r)

        if score > best_score:
            best_score = score
            best_thresh = float(thresh)

    logger.info(f"Optimal threshold ({metric}, beta={beta}): {best_thresh:.4f} with score: {best_score:.4f}")
    return best_thresh, float(best_score)


def calculate_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Compute full evaluation metrics suite for a model at a given threshold.
    """
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    # [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()

    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    f2 = float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0))

    try:
        avg_precision = float(average_precision_score(y_true, y_prob))
    except Exception:
        avg_precision = 0.0

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.0

    results = {
        "model_name": model_name,
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "f2_score": f2,
        "average_precision": avg_precision,
        "pr_auc": avg_precision,
        "roc_auc": roc_auc,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "total_test_samples": int(len(y_true)),
        "actual_positives": int(tp + fn)
    }

    return results


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    model_name: str,
    output_path: str
):
    """Generate and save confusion matrix heatmap."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate (0)", "Laundering (1)"],
        yticklabels=["Legitimate (0)", "Laundering (1)"]
    )
    plt.title(f"Confusion Matrix - {model_name}\n(Threshold: {threshold:.4f})")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix plot to {output_path}")


def plot_curves(
    eval_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],  # {model_name: (y_true, y_prob)}
    pr_curve_path: str,
    roc_curve_path: str
):
    """
    Generate and save Precision-Recall and ROC curves comparing multiple models.
    """
    # Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    for name, (y_true, y_prob) in eval_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        plt.plot(recall, precision, lw=2, label=f"{name} (PR-AUC / AP = {ap:.4f})")

    plt.xlabel("Recall (Detection Rate)")
    plt.ylabel("Precision (Accuracy of Flags)")
    plt.title("Precision-Recall Curves — Model Comparison")
    plt.legend(loc="lower left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    os.makedirs(os.path.dirname(pr_curve_path), exist_ok=True)
    plt.savefig(pr_curve_path, dpi=300)
    plt.close()
    logger.info(f"Saved Precision-Recall curve to {pr_curve_path}")

    # ROC Curve
    plt.figure(figsize=(8, 6))
    for name, (y_true, y_prob) in eval_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (ROC-AUC = {auc:.4f})")

    plt.plot([0, 1], [0, 1], color="grey", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("Receiver Operating Characteristic (ROC) Curves")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    os.makedirs(os.path.dirname(roc_curve_path), exist_ok=True)
    plt.savefig(roc_curve_path, dpi=300)
    plt.close()
    logger.info(f"Saved ROC curve to {roc_curve_path}")
