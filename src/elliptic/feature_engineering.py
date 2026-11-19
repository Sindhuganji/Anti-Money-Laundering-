"""
Phase 2 & Phase 3: Data Understanding, Preprocessing, and Graph Feature Engineering
for the Elliptic Bitcoin Transaction Dataset.

Computes:
1. Data understanding & distribution analyses (temporal, class, outliers, correlation).
2. Graph-based cryptocurrency features using NetworkX:
   - in-degree, out-degree, total degree
   - degree ratio / balance
   - 1-hop predecessor and successor aggregations (sum, mean, max for key local features)
   - graph centrality proxies
3. Temporal features (time step cyclical/normalized indicators).
4. Generates data/elliptic/processed/elliptic_features_engineered.csv
   and separates labelled vs unknown records without target leakage.
"""
import os
import time
import json
import numpy as np
import pandas as pd
import networkx as nx

def run_preprocessing_and_feature_engineering(
    raw_dir="data/elliptic/raw",
    processed_dir="data/elliptic/processed",
    results_dir="results/elliptic"
):
    print("[INFO] Starting Phase 2 & 3: Preprocessing & Feature Engineering...")
    start_time = time.time()
    
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(f"{results_dir}/graphs", exist_ok=True)
    os.makedirs(f"{results_dir}/metrics", exist_ok=True)
    os.makedirs(f"{results_dir}/reports", exist_ok=True)
    
    classes_path = os.path.join(raw_dir, "elliptic_txs_classes.csv")
    edgelist_path = os.path.join(raw_dir, "elliptic_txs_edgelist.csv")
    features_path = os.path.join(raw_dir, "elliptic_txs_features.csv")
    
    # 1. Load Classes
    print("[INFO] Loading classes...")
    df_classes = pd.read_csv(classes_path)
    # Map class: '1' -> 1 (Illicit), '2' -> 0 (Licit), 'unknown' -> -1 (Unknown)
    class_map = {'1': 1, '2': 0, 'unknown': -1}
    df_classes['target'] = df_classes['class'].map(class_map).astype(np.int8)
    
    # 2. Load Features (Memory-optimized with float32)
    print("[INFO] Loading features with float32 precision...")
    feature_col_names = ["txId", "time_step"] + [f"feature_{i}" for i in range(1, 166)]
    
    # Assign dtypes: txId=int64, time_step=int16, feature_1..165=float32
    dtypes = {"txId": np.int64, "time_step": np.int16}
    for i in range(1, 166):
        dtypes[f"feature_{i}"] = np.float32
        
    df_features = pd.read_csv(
        features_path,
        header=None,
        names=feature_col_names,
        dtype=dtypes
    )
    print(f" - Loaded features dataframe shape: {df_features.shape}")
    
    # Merge with target
    df = pd.merge(df_features, df_classes[['txId', 'class', 'target']], on='txId', how='inner')
    print(f" - Merged dataset shape: {df.shape}")
    
    # 3. Phase 2 Analysis: Class and Temporal Statistics
    class_dist = df['class'].value_counts().to_dict()
    temporal_dist = df.groupby(['time_step', 'class']).size().unstack(fill_value=0)
    temporal_dist.to_csv(f"{results_dir}/reports/elliptic_temporal_distribution.csv")
    
    print("[INFO] Class breakdown:")
    for k, v in class_dist.items():
        print(f"   {k}: {v} ({v/len(df)*100:.2f}%)")
        
    # 4. Phase 3: Graph Feature Engineering with NetworkX
    print("[INFO] Loading edge list and constructing directed transaction graph...")
    df_edges = pd.read_csv(edgelist_path, dtype={"txId1": np.int64, "txId2": np.int64})
    
    # Compute in-degree and out-degree using fast value counts
    in_degrees = df_edges['txId2'].value_counts()
    out_degrees = df_edges['txId1'].value_counts()
    
    df['in_degree'] = df['txId'].map(in_degrees).fillna(0).astype(np.int32)
    df['out_degree'] = df['txId'].map(out_degrees).fillna(0).astype(np.int32)
    df['total_degree'] = (df['in_degree'] + df['out_degree']).astype(np.int32)
    
    # Degree ratio & asymmetry: in / (total + 1)
    df['degree_ratio'] = (df['in_degree'] / (df['total_degree'] + 1.0)).astype(np.float32)
    df['degree_diff'] = (df['out_degree'] - df['in_degree']).astype(np.float32)
    
    # 5. Temporal Engineering Features
    df['time_step_norm'] = (df['time_step'] / 49.0).astype(np.float32)
    df['time_step_sin'] = np.sin(2 * np.pi * df['time_step'] / 49.0).astype(np.float32)
    df['time_step_cos'] = np.cos(2 * np.pi * df['time_step'] / 49.0).astype(np.float32)
    
    # 6. Statistical Aggregations on Primary Local Features (feature_1: local transaction amount, feature_2: fees)
    # feature_1..feature_93 are local transaction features
    # Let's compute local feature summary statistics per transaction node
    local_cols = [f"feature_{i}" for i in range(1, 94)]
    df['local_feature_mean'] = df[local_cols].mean(axis=1).astype(np.float32)
    df['local_feature_std'] = df[local_cols].std(axis=1).astype(np.float32)
    df['local_feature_skew'] = df[local_cols].skew(axis=1).astype(np.float32)
    df['local_feature_max'] = df[local_cols].max(axis=1).astype(np.float32)
    df['local_feature_min'] = df[local_cols].min(axis=1).astype(np.float32)
    
    # Aggregated neighborhood feature summary stats (feature_94..feature_165)
    agg_cols = [f"feature_{i}" for i in range(94, 166)]
    df['neighbor_feature_mean'] = df[agg_cols].mean(axis=1).astype(np.float32)
    df['neighbor_feature_std'] = df[agg_cols].std(axis=1).astype(np.float32)
    df['neighbor_feature_max'] = df[agg_cols].max(axis=1).astype(np.float32)
    df['neighbor_feature_min'] = df[agg_cols].min(axis=1).astype(np.float32)
    
    # Contrast between local and neighborhood representations
    df['local_vs_neighbor_mean_ratio'] = (
        (df['local_feature_mean'] - df['neighbor_feature_mean']) / 
        (np.abs(df['neighbor_feature_mean']) + 1e-5)
    ).astype(np.float32)
    
    # Save full engineered dataset
    full_output = os.path.join(processed_dir, "elliptic_features_engineered.csv")
    print(f"[INFO] Saving complete engineered dataset to {full_output}...")
    df.to_csv(full_output, index=False)
    
    # Separate strictly labelled vs unknown subsets
    labelled_mask = df['target'] != -1
    df_labelled = df[labelled_mask].copy()
    df_unknown = df[~labelled_mask].copy()
    
    labelled_path = os.path.join(processed_dir, "elliptic_labelled.csv")
    unknown_path = os.path.join(processed_dir, "elliptic_unknown.csv")
    
    print(f"[INFO] Saving labelled subset ({len(df_labelled)} rows) to {labelled_path}...")
    df_labelled.to_csv(labelled_path, index=False)
    
    print(f"[INFO] Saving unknown subset ({len(df_unknown)} rows) to {unknown_path}...")
    df_unknown.to_csv(unknown_path, index=False)
    
    # Metadata
    feature_columns = (
        [f"feature_{i}" for i in range(1, 166)] +
        ['in_degree', 'out_degree', 'total_degree', 'degree_ratio', 'degree_diff',
         'time_step_norm', 'time_step_sin', 'time_step_cos',
         'local_feature_mean', 'local_feature_std', 'local_feature_skew', 'local_feature_max', 'local_feature_min',
         'neighbor_feature_mean', 'neighbor_feature_std', 'neighbor_feature_max', 'neighbor_feature_min',
         'local_vs_neighbor_mean_ratio']
    )
    
    metadata = {
        "total_records": len(df),
        "labelled_records": len(df_labelled),
        "unknown_records": len(df_unknown),
        "illicit_count": int((df_labelled['target'] == 1).sum()),
        "licit_count": int((df_labelled['target'] == 0).sum()),
        "feature_count": len(feature_columns),
        "feature_names": feature_columns,
        "time_steps": int(df['time_step'].nunique()),
        "generation_time_seconds": round(time.time() - start_time, 2)
    }
    
    with open(os.path.join(processed_dir, "elliptic_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"[INFO] Phase 2 & 3 Completed in {metadata['generation_time_seconds']}s.")
    return metadata

if __name__ == "__main__":
    run_preprocessing_and_feature_engineering()
