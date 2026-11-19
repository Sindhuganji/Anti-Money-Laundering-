"""
Phase 7: Cryptocurrency AML Risk Scoring Engine.
Maps model illicit probabilities and network context to a calibrated 0-100 risk score:
- Low Risk (0 - 29)
- Medium Risk (30 - 69)
- High Risk (70 - 89)
- Critical Risk (90 - 100)

Explicitly handles:
- Model probability calibration
- Graph degree centrality risk penalty
- Unknown transaction flag and handling
- Compliance disclaimer (not legal proof)
"""
import numpy as np

def calculate_crypto_risk_score(probability: float, degree_ratio: float = 0.5, total_degree: int = 0) -> float:
    """
    Computes a composite crypto AML risk score between 0.0 and 100.0.
    
    Parameters:
    - probability: Model predicted probability of being an illicit transaction (0.0 to 1.0).
    - degree_ratio: in_degree / (total_degree + 1.0).
    - total_degree: total transaction graph connections.
    
    Returns:
    - risk_score: float in [0.0, 100.0]
    """
    prob_clamped = float(np.clip(probability, 0.0, 1.0))
    base_score = prob_clamped * 100.0
    
    # Network context boost:
    # High fan-out / peeling chains (high out-degree, low degree ratio) or high clustering
    network_penalty = 0.0
    if total_degree >= 10:
        network_penalty += min(5.0, total_degree * 0.2)
    if degree_ratio > 0.85 or degree_ratio < 0.15:
        network_penalty += 3.0  # Asymmetric flow pattern typical of mixing/peeling
        
    final_score = float(np.clip(base_score + network_penalty, 0.0, 100.0))
    return round(final_score, 2)

def assign_crypto_risk_tier(risk_score: float) -> str:
    """Assigns risk tier according to standard compliance boundaries."""
    if risk_score >= 90.0:
        return "Critical"
    elif risk_score >= 70.0:
        return "High"
    elif risk_score >= 30.0:
        return "Medium"
    else:
        return "Low"

def format_crypto_risk_assessment(
    tx_id: int,
    probability: float,
    threshold: float,
    total_degree: int = 0,
    degree_ratio: float = 0.5,
    actual_label: str = "unknown"
) -> dict:
    """Generates structured compliance risk assessment dictionary."""
    is_illicit_predicted = bool(probability >= threshold)
    score = calculate_crypto_risk_score(probability, degree_ratio, total_degree)
    tier = assign_crypto_risk_tier(score)
    
    is_unknown = (actual_label == "unknown" or actual_label == "-1")
    
    return {
        "txId": tx_id,
        "Illicit_Probability": round(float(probability), 4),
        "Decision_Threshold": round(float(threshold), 4),
        "Predicted_Class": "Illicit" if is_illicit_predicted else "Licit",
        "Risk_Score": score,
        "Risk_Tier": tier,
        "Network_Connections": total_degree,
        "Ground_Truth_Class": actual_label,
        "Is_Ground_Truth_Unknown": is_unknown,
        "Compliance_Disclaimer": "Cryptocurrency risk scores and machine learning predictions are indicators for compliance triage and enhanced due diligence; they do not constitute definitive legal proof of illicit activity."
    }
