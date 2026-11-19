"""
Model training and threshold optimization pipeline for AML detection.
Trains:
1. Random Forest (with balanced_subsample class weights, n_estimators=100, max_depth=14)
2. XGBoost (with scale_pos_weight tuned to account for class imbalance, max_depth=6, learning_rate=0.08)

Optimizes classification thresholds strictly on validation data to maximize F2 / PR trade-off.
Evaluates final unbiased metrics on test data.
Persists trained models, thresholds, metrics, and generates comparison plots.
"""

import os
import sys
import json
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.preprocessing import AMLPreprocessingPipeline
from src.evaluation import (
    find_optimal_threshold,
    calculate_metrics,
    plot_confusion_matrix,
    plot_curves
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = "models"
RESULTS_METRICS_DIR = os.path.join("results", "metrics")
RESULTS_GRAPHS_DIR = os.path.join("results", "graphs")
TRAIN_DATA_PATH = os.path.join("data", "processed", "aml_train.csv")
VAL_DATA_PATH = os.path.join("data", "processed", "aml_val.csv")
TEST_DATA_PATH = os.path.join("data", "processed", "aml_test.csv")


def load_and_preprocess_splits(
    max_train_samples: int = 300000,
    max_val_samples: int = 100000,
    max_test_samples: int = None,
    random_state: int = 42
):
    """
    Load data splits and transform using the pre-fitted AMLPreprocessingPipeline.
    Uses all positive cases in training + stratified sample of negative cases up to max_train_samples
    if dataset is large, preserving genuine positive transaction frequency.
    """
    logger.info("Loading preprocessing pipeline...")
    pipeline = AMLPreprocessingPipeline.load(model_dir=MODELS_DIR, prefix="aml_pipeline")
    feature_names = pipeline.feature_names

    logger.info("Loading training data...")
    df_train = pd.read_csv(TRAIN_DATA_PATH)
    if max_train_samples and len(df_train) > max_train_samples:
        pos_df = df_train[df_train['Is Laundering'] == 1]
        neg_df = df_train[df_train['Is Laundering'] == 0].sample(
            n=max_train_samples - len(pos_df),
            random_state=random_state
        )
        df_train = pd.concat([pos_df, neg_df]).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        logger.info(f"Sampled training set: {df_train.shape} (positives: {df_train['Is Laundering'].sum()})")

    logger.info("Loading validation data...")
    df_val = pd.read_csv(VAL_DATA_PATH)
    if max_val_samples and len(df_val) > max_val_samples:
        pos_val = df_val[df_val['Is Laundering'] == 1]
        neg_val = df_val[df_val['Is Laundering'] == 0].sample(
            n=max_val_samples - len(pos_val),
            random_state=random_state
        )
        df_val = pd.concat([pos_val, neg_val]).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        logger.info(f"Sampled validation set: {df_val.shape} (positives: {df_val['Is Laundering'].sum()})")

    logger.info("Loading test data...")
    df_test = pd.read_csv(TEST_DATA_PATH)
    if max_test_samples and len(df_test) > max_test_samples:
        pos_test = df_test[df_test['Is Laundering'] == 1]
        neg_test = df_test[df_test['Is Laundering'] == 0].sample(
            n=max_test_samples - len(pos_test),
            random_state=random_state
        )
        df_test = pd.concat([pos_test, neg_test]).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        logger.info(f"Sampled test set: {df_test.shape} (positives: {df_test['Is Laundering'].sum()})")

    y_train = df_train['Is Laundering'].to_numpy(dtype=np.int8)
    y_val = df_val['Is Laundering'].to_numpy(dtype=np.int8)
    y_test = df_test['Is Laundering'].to_numpy(dtype=np.int8)

    logger.info("Transforming feature matrices with pipeline...")
    X_train = pipeline.transform(df_train)
    X_val = pipeline.transform(df_val)
    X_test = pipeline.transform(df_test)

    return X_train, y_train, X_val, y_val, X_test, y_test, pipeline


def train_random_forest(X_train, y_train, random_state=42):
    """
    Train Random Forest with balanced subsampling to handle extreme class imbalance.
    """
    logger.info("Training Random Forest Classifier...")
    rf = RandomForestClassifier(
        n_estimators=120,
        max_depth=14,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    logger.info("Random Forest training completed.")
    return rf


def train_xgboost(X_train, y_train, random_state=42):
    """
    Train XGBoost with scale_pos_weight to heavily penalize false negatives on positive cases.
    """
    neg_count = int(np.sum(y_train == 0))
    pos_count = int(np.sum(y_train == 1))
    scale_pos_weight = float(neg_count / max(1, pos_count))
    logger.info(f"Training XGBoost Classifier with scale_pos_weight={scale_pos_weight:.2f}...")

    xgb = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=random_state,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    logger.info("XGBoost training completed.")
    return xgb


def run_training_pipeline(
    max_train_samples: int = 300000,
    max_val_samples: int = 100000,
    max_test_samples: int = None
):
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_METRICS_DIR, exist_ok=True)
    os.makedirs(RESULTS_GRAPHS_DIR, exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test, pipeline = load_and_preprocess_splits(
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        max_test_samples=max_test_samples
    )

    # ------------------ RANDOM FOREST ------------------
    rf_model = train_random_forest(X_train, y_train)
    rf_val_prob = rf_model.predict_proba(X_val)[:, 1]
    rf_best_thresh, rf_val_score = find_optimal_threshold(y_val, rf_val_prob, metric="f2")

    rf_test_prob = rf_model.predict_proba(X_test)[:, 1]
    rf_metrics = calculate_metrics(y_test, rf_test_prob, threshold=rf_best_thresh, model_name="Random Forest")
    rf_metrics_default = calculate_metrics(y_test, rf_test_prob, threshold=0.5, model_name="Random Forest (Default 0.5)")

    logger.info(f"RF Test Metrics at optimal threshold ({rf_best_thresh:.4f}):\n{json.dumps(rf_metrics, indent=2)}")

    # Save RF artifacts
    joblib.dump(rf_model, os.path.join(MODELS_DIR, "random_forest_model.pkl"))
    with open(os.path.join(MODELS_DIR, "random_forest_threshold.json"), "w") as f:
        json.dump({"optimal_threshold": rf_best_thresh, "optimization_metric": "f2"}, f, indent=4)
    plot_confusion_matrix(
        y_test, rf_test_prob, rf_best_thresh,
        "Random Forest",
        os.path.join(RESULTS_GRAPHS_DIR, "rf_confusion_matrix.png")
    )

    # ------------------ XGBOOST ------------------
    xgb_model = train_xgboost(X_train, y_train)
    xgb_val_prob = xgb_model.predict_proba(X_val)[:, 1]
    xgb_best_thresh, xgb_val_score = find_optimal_threshold(y_val, xgb_val_prob, metric="f2")

    xgb_test_prob = xgb_model.predict_proba(X_test)[:, 1]
    xgb_metrics = calculate_metrics(y_test, xgb_test_prob, threshold=xgb_best_thresh, model_name="XGBoost")
    xgb_metrics_default = calculate_metrics(y_test, xgb_test_prob, threshold=0.5, model_name="XGBoost (Default 0.5)")

    logger.info(f"XGB Test Metrics at optimal threshold ({xgb_best_thresh:.4f}):\n{json.dumps(xgb_metrics, indent=2)}")

    # Save XGB artifacts
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, "xgboost_model.pkl"))
    with open(os.path.join(MODELS_DIR, "xgboost_threshold.json"), "w") as f:
        json.dump({"optimal_threshold": xgb_best_thresh, "optimization_metric": "f2"}, f, indent=4)
    plot_confusion_matrix(
        y_test, xgb_test_prob, xgb_best_thresh,
        "XGBoost",
        os.path.join(RESULTS_GRAPHS_DIR, "xgb_confusion_matrix.png")
    )

    # ------------------ COMPARISON & CURVES ------------------
    comparison_df = pd.DataFrame([rf_metrics_default, rf_metrics, xgb_metrics_default, xgb_metrics])
    comparison_path = os.path.join(RESULTS_METRICS_DIR, "model_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)
    logger.info(f"Model comparison saved to {comparison_path}")

    # Plot Curves
    eval_dict = {
        "Random Forest": (y_test, rf_test_prob),
        "XGBoost": (y_test, xgb_test_prob)
    }
    plot_curves(
        eval_dict,
        pr_curve_path=os.path.join(RESULTS_GRAPHS_DIR, "precision_recall_curves.png"),
        roc_curve_path=os.path.join(RESULTS_GRAPHS_DIR, "roc_curves.png")
    )

    # Save feature names list
    with open(os.path.join(MODELS_DIR, "feature_names.json"), "w") as f:
        json.dump(pipeline.feature_names, f, indent=4)

    logger.info("Training, validation thresholding, and testing pipeline completed successfully!")


if __name__ == "__main__":
    run_training_pipeline()
