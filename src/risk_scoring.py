"""
Risk scoring and category classification engine for AML detection.
Maps model output probabilities (0.0 - 1.0) into standardized 0 - 100 risk scores
and discrete financial alert risk tiers:
- Low Risk (0 - 29)
- Medium Risk (30 - 59)
- High Risk (60 - 79)
- Critical Risk (80 - 100)
"""

import json
import logging
from typing import Dict, Any, Tuple, Union
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_RISK_THRESHOLDS = {
    "Low": (0, 29),
    "Medium": (30, 59),
    "High": (60, 79),
    "Critical": (80, 100)
}


def calculate_risk_score(
    probability: Union[float, np.ndarray],
    scale: float = 100.0,
    calibrated_threshold: float = 0.5
) -> Union[float, np.ndarray]:
    """
    Convert model probability into an intuitive 0-100 AML Risk Score.
    
    If probability is below the optimal detection threshold, scales smoothly from 0 to 59.
    If probability is at or above the optimal detection threshold, scales from 60 to 100.
    """
    if isinstance(probability, (list, np.ndarray)):
        prob_arr = np.asarray(probability, dtype=float)
        scores = np.where(
            prob_arr < calibrated_threshold,
            (prob_arr / max(1e-6, calibrated_threshold)) * 59.0,
            60.0 + ((prob_arr - calibrated_threshold) / max(1e-6, (1.0 - calibrated_threshold))) * 40.0
        )
        return np.clip(np.round(scores, 1), 0.0, 100.0)
    else:
        prob = float(probability)
        if prob < calibrated_threshold:
            score = (prob / max(1e-6, calibrated_threshold)) * 59.0
        else:
            score = 60.0 + ((prob - calibrated_threshold) / max(1e-6, (1.0 - calibrated_threshold))) * 40.0
        return float(np.clip(round(score, 1), 0.0, 100.0))


def assign_risk_category(
    risk_score: Union[float, int]
) -> Tuple[str, str]:
    """
    Assign risk level and color code based on score:
    - 0-29: Low Risk (Green)
    - 30-59: Medium Risk (Amber)
    - 60-79: High Risk (Orange)
    - 80-100: Critical Risk (Red)
    """
    score = float(risk_score)
    if score >= 80.0:
        return "Critical Risk", "#DC2626"  # Red
    elif score >= 60.0:
        return "High Risk", "#EA580C"  # Orange
    elif score >= 30.0:
        return "Medium Risk", "#F59E0B"  # Amber
    else:
        return "Low Risk", "#10B981"  # Green


def evaluate_transaction_risk(
    probability: float,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluate single transaction probability into full risk object.
    """
    is_suspicious = bool(probability >= threshold)
    risk_score = calculate_risk_score(probability, calibrated_threshold=threshold)
    category, color = assign_risk_category(risk_score)

    return {
        "probability": float(probability),
        "threshold": float(threshold),
        "is_suspicious": is_suspicious,
        "risk_score": float(risk_score),
        "risk_category": category,
        "risk_color": color,
        "investigation_recommended": bool(risk_score >= 60.0)
    }
