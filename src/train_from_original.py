"""Train IBM AML models from data/raw/HI-Small_Trans.csv.
Run: ./.venv/bin/python -m src.train_from_original
Optional: --max-train-rows 500000 for faster experimentation (train only).
"""
import os, json, logging, argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from src.data_loader import load_raw_data, validate_raw_schema
from src.preprocessing import AMLPreprocessingPipeline, chronological_split
from src.feature_engineering import (extract_temporal_features, extract_amount_features,
    extract_relational_features, extract_historical_features, fit_frequency_encoders,
    apply_frequency_encoders, ENGINEERED_FEATURE_NAMES)
from src.evaluation import find_optimal_threshold, calculate_metrics, plot_confusion_matrix, plot_curves

RAW_DATA_PATH = "data/raw/HI-Small_Trans.csv"
TARGET = "Is Laundering"
MODELS_DIR, PROCESSED_DIR = "models", "data/processed"
METRICS_DIR, GRAPHS_DIR = "results/metrics", "results/graphs"
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("aml_training")


def normalize_target(df):
    out = df.copy(); y = pd.to_numeric(out[TARGET], errors="coerce")
    if y.isna().any() or not set(y.unique()).issubset({0, 1}): raise ValueError("Target must contain only non-missing 0/1 values.")
    out[TARGET] = y.astype("int8"); return out


def engineer_raw_splits(raw):
    # Split and process one chronologically ordered stream. Historical values
    # for validation/test may use earlier partitions, never later transactions.
    ordered = raw.copy().reset_index(drop=True)
    ordered["__time"] = pd.to_datetime(ordered["Timestamp"], errors="coerce")
    ordered["__order"] = np.arange(len(ordered))
    ordered = ordered.sort_values(["__time", "__order"], kind="mergesort").drop(columns=["__time", "__order"]).reset_index(drop=True)
    train_raw, val_raw, test_raw = chronological_split(ordered)
    n_train, n_val = len(train_raw), len(val_raw)
    feat = extract_temporal_features(ordered)
    feat = extract_amount_features(feat)
    feat = extract_relational_features(feat)
    feat = extract_historical_features(feat)
    encoders = fit_frequency_encoders(ordered.iloc[:n_train])
    feat = apply_frequency_encoders(feat, encoders)
    missing = sorted(set(ENGINEERED_FEATURE_NAMES) - set(feat.columns))
    if missing: raise ValueError(f"Missing engineered features: {missing}")
    engineered = feat[ENGINEERED_FEATURE_NAMES].copy()
    engineered[TARGET] = ordered[TARGET].to_numpy()
    a, b = n_train, n_train + n_val
    return (engineered.iloc[:a].reset_index(drop=True), engineered.iloc[a:b].reset_index(drop=True),
            engineered.iloc[b:].reset_index(drop=True), encoders)


def sample_training_rows(df, max_rows=None, seed=42):
    if max_rows is None or len(df) <= max_rows: return df
    positives, negatives = df[df[TARGET] == 1], df[df[TARGET] == 0]
    if len(positives) >= max_rows: return df
    sampled_neg = negatives.sample(n=min(len(negatives), max_rows-len(positives)), random_state=seed)
    return pd.concat([positives, sampled_neg]).sample(frac=1, random_state=seed).reset_index(drop=True)


def train_random_forest(X, y, seed=42):
    model = RandomForestClassifier(n_estimators=350, max_depth=16, min_samples_split=8,
        min_samples_leaf=2, max_features="sqrt", class_weight="balanced_subsample",
        random_state=seed, n_jobs=-1)
    model.fit(X, y); return model


def train_xgboost(X, y, seed=42):
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if pos == 0: raise ValueError("No positive laundering examples in training set.")
    model = XGBClassifier(n_estimators=700, max_depth=4, learning_rate=0.03,
        min_child_weight=3, subsample=0.8, colsample_bytree=0.8, gamma=0.05,
        reg_alpha=0.1, reg_lambda=5.0, scale_pos_weight=neg/pos,
        objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
        random_state=seed, n_jobs=-1)
    model.fit(X, y, verbose=False); return model


def run_training(raw_path=RAW_DATA_PATH, max_train_rows=None, random_state=42):
    for d in (MODELS_DIR, PROCESSED_DIR, METRICS_DIR, GRAPHS_DIR): os.makedirs(d, exist_ok=True)
    log.info("Loading original dataset: %s", raw_path)
    raw = load_raw_data(file_path=raw_path, drop_duplicates=True)
    validate_raw_schema(raw); raw = normalize_target(raw)
    train_df, val_df, test_df, encoders = engineer_raw_splits(raw)
    for name, part in zip(("Train", "Validation", "Test"), (train_df, val_df, test_df)):
        if int(part[TARGET].sum()) == 0:
            raise ValueError(f"{name} partition has no positive examples.")
    train_df = sample_training_rows(train_df, max_train_rows, random_state)
    train_df.to_csv(os.path.join(PROCESSED_DIR, "aml_train.csv"), index=False)
    val_df.to_csv(os.path.join(PROCESSED_DIR, "aml_val.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR, "aml_test.csv"), index=False)
    y_train, y_val, y_test = [x[TARGET].to_numpy(dtype=np.int8) for x in (train_df, val_df, test_df)]
    X_train_raw, X_val_raw, X_test_raw = [x.drop(columns=TARGET) for x in (train_df, val_df, test_df)]
    pipeline = AMLPreprocessingPipeline("robust", "median")
    X_train = pipeline.fit_transform(X_train_raw, encoders)
    X_val, X_test = pipeline.transform(X_val_raw), pipeline.transform(X_test_raw)
    pipeline.save(MODELS_DIR, "aml_pipeline")
    models = {"Random Forest": train_random_forest(X_train, y_train, random_state),
              "XGBoost": train_xgboost(X_train, y_train, random_state)}
    rows, curves = [], {}
    for name, model in models.items():
        key = name.lower().replace(" ", "_")
        val_prob = model.predict_proba(X_val)[:, 1]
        threshold, val_f2 = find_optimal_threshold(y_val, val_prob, metric="f2")
        test_prob = model.predict_proba(X_test)[:, 1]
        metrics = calculate_metrics(y_test, test_prob, threshold=threshold, model_name=name)
        metrics["validation_f2"] = float(val_f2); rows.append(metrics)
        rows.append(calculate_metrics(y_test, test_prob, threshold=.5, model_name=f"{name} (0.5)"))
        curves[name] = (y_test, test_prob)
        joblib.dump(model, os.path.join(MODELS_DIR, f"{key}_model.pkl"))
        with open(os.path.join(MODELS_DIR, f"{key}_threshold.json"), "w") as f:
            json.dump({"optimal_threshold": float(threshold), "optimization_metric": "f2",
                       "validation_f2": float(val_f2), "threshold_selected_on": "validation"}, f, indent=2)
        plot_confusion_matrix(y_test, test_prob, threshold, name, os.path.join(GRAPHS_DIR, f"{key}_confusion_matrix.png"))
        log.info("%s: validation F2=%.4f threshold=%.6f", name, val_f2, threshold)
    pd.DataFrame(rows).to_csv(os.path.join(METRICS_DIR, "model_comparison.csv"), index=False)
    plot_curves(curves, os.path.join(GRAPHS_DIR, "precision_recall_curves.png"), os.path.join(GRAPHS_DIR, "roc_curves.png"))
    with open(os.path.join(MODELS_DIR, "feature_names.json"), "w") as f: json.dump(pipeline.feature_names, f, indent=2)
    with open(os.path.join(MODELS_DIR, "aml_training_metadata.json"), "w") as f:
        json.dump({"source_dataset": raw_path, "split_strategy": "chronological 64/16/20",
                   "feature_names": pipeline.feature_names, "train_rows": len(train_df),
                   "validation_rows": len(val_df), "test_rows": len(test_df),
                   "train_positive": int(y_train.sum()), "validation_positive": int(y_val.sum()),
                   "test_positive": int(y_test.sum())}, f, indent=2)
    log.info("Training complete. Thresholds selected on validation; final metrics evaluated on test.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data", default=RAW_DATA_PATH)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(); run_training(args.raw_data, args.max_train_rows, args.seed)
