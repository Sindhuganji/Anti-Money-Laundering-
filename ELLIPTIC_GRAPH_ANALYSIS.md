# Elliptic Bitcoin Transaction Network Graph Analysis Report

## Executive Summary
This report analyzes the topological structure and payment-flow mechanics of the **Elliptic Bitcoin Transaction Graph**. Bitcoin transactions inherently form a Directed Acyclic Graph (DAG) where outputs of ancestor transactions become inputs to descendant transactions.

---

## 1. Network Topology Metrics
- **Total Directed Edges**: 234,355
- **Total Participating Nodes**: 203,769
- **Sample Visualization Graph**:
  - Subgraph Nodes: 260
  - Subgraph Edges: 250
  - Density: 0.0037

---

## 2. Graph Structural Patterns Observed
1. **Peeling Chains & Dispersal Trees**: Illicit entities frequently utilize high out-degree transactions with small change amounts returning to new addresses, creating chain-like branching patterns designed to evade volume thresholds.
2. **Consolidation / Fan-In Funnels**: Prior to cashing out at exchanges or darknet marketplaces, multiple smaller UTXOs are funneled into high in-degree transaction hubs.
3. **Neighborhood Homophily**: Illicit transactions cluster tightly with other illicit or unknown nodes in localized network neighborhoods, validating the predictive efficacy of 1-hop aggregated features (`neighbor_feature_mean`, `in_degree`, `degree_ratio`).

---

## 3. Visualization Artifact
- **Interactive Network Visualization**: `results/elliptic/graphs/transaction_network.html`
- Red nodes: Confirmed Illicit transactions (Class 1)
- Green nodes: Confirmed Licit transactions (Class 2)
- Gray nodes: Unlabelled / Unknown transactions
