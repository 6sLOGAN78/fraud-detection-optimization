"""Pipeline script to execute Part 17 — Advanced Fraud Modeling, Graph & Deep Learning, Streaming & Federated AI."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.models import (
    AdvancedStackingMetaLearner,
    AutoMLEngine,
    ContinuousOnlineLearner,
    FederatedLearningAggregator,
    FraudRLPolicyAgent,
    GNNFraudDetector,
    SelfSupervisedAutoencoder,
    SemiSupervisedPseudoLabeler,
    StreamingInferenceEngine,
    TabularDeepMLP,
    TabularTransformerModel,
    TransactionGraphEngine,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Part 17 Advanced AI & Modeling Pipeline")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of samples to process")
    args = parser.parse_args()

    train_path = Path("data/interim/train_cleaned.parquet")
    if not train_path.exists():
        train_path = Path("data/interim/train_merged.parquet")

    logger.info(f"Loading dataset from {train_path}...")
    df = pd.read_parquet(train_path)

    if len(df) > args.n_samples:
        df = df.sample(n=args.n_samples, random_state=42).reset_index(drop=True)

    y = df["isFraud"].values if "isFraud" in df.columns else np.zeros(len(df))
    X_num = df.select_dtypes(include=[np.number]).fillna(0)

    # 17.1 Advanced Stacking
    logger.info("Executing 17.1 Advanced Stacking Meta-Learner...")
    b1 = RandomForestClassifier(n_estimators=10, random_state=42)
    b2 = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=43)
    stacker = AdvancedStackingMetaLearner(base_models=[b1, b2], n_splits=3)
    stacker.fit(X_num, y)
    stack_probs = stacker.predict_proba(X_num)[:, 1]

    # 17.2 Semi-Supervised Pseudo-Labeling
    logger.info("Executing 17.2 Semi-Supervised Pseudo-Labeler...")
    pseudo_labeler = SemiSupervisedPseudoLabeler(high_confidence_threshold=0.8, low_confidence_threshold=0.2)
    X_pseudo, y_pseudo = pseudo_labeler.generate_pseudo_labels(b1, X_num)

    # 17.3 Self-Supervised Autoencoder
    logger.info("Executing 17.3 Self-Supervised Autoencoder...")
    autoencoder = SelfSupervisedAutoencoder(input_dim=X_num.shape[1], hidden_dim=16)
    autoencoder.fit(X_num, epochs=3)
    recon_errors = autoencoder.compute_reconstruction_error(X_num)

    # 17.4 & 17.7 Graph Engine & GNN Risk Detector
    logger.info("Executing 17.4 & 17.7 Graph Engine & GNN Detector...")
    graph_engine = TransactionGraphEngine()
    df_graph = graph_engine.compute_graph_metrics(X_num)
    gnn = GNNFraudDetector(input_dim=X_num.shape[1])
    gnn.fit(X_num, y)
    gnn_risks = gnn.predict_node_risk(X_num)

    # 17.5 & 17.6 Tabular Deep Learning & Transformer
    logger.info("Executing 17.5 & 17.6 Deep Learning & Transformer Models...")
    mlp = TabularDeepMLP(input_dim=X_num.shape[1])
    mlp.fit(X_num, y)
    transformer = TabularTransformerModel(input_dim=X_num.shape[1])
    transformer.fit(X_num, y)

    # 17.8 Streaming Inference Engine
    logger.info("Executing 17.8 Real-Time Streaming Inference Engine...")
    stream_engine = StreamingInferenceEngine(model=b1, feature_names=list(X_num.columns))
    sample_events = [X_num.iloc[0].to_dict(), X_num.iloc[1].to_dict()]
    stream_res = stream_engine.process_event_stream(sample_events)

    # 17.9 Federated Learning Aggregator
    logger.info("Executing 17.9 Federated Learning Aggregator...")
    fed_agg = FederatedLearningAggregator()
    w1 = np.ones((X_num.shape[1], 1))
    w2 = np.ones((X_num.shape[1], 1)) * 2
    fed_weights = fed_agg.aggregate_client_weights([w1, w2])

    # 17.10 AutoML Integration
    logger.info("Executing 17.10 AutoML Search...")
    automl = AutoMLEngine()
    automl_res = automl.run_automl_search(X_num, y)

    # 17.11 Reinforcement Learning Agent
    logger.info("Executing 17.11 Reinforcement Learning Policy Agent...")
    rl_agent = FraudRLPolicyAgent(initial_threshold=0.5)
    new_thresh = rl_agent.select_action_threshold(recent_chargeback_rate=0.06)

    # 17.12 Continuous Online Learner
    logger.info("Executing 17.12 Continuous Online Learner...")
    online_learner = ContinuousOnlineLearner(feature_names=list(X_num.columns))
    online_prob = online_learner.update_online(X_num.iloc[0].to_dict(), y_true=1)

    summary_report = {
        "stacking_mean_probability": round(float(np.mean(stack_probs)), 4),
        "pseudo_labels_generated": len(X_pseudo),
        "mean_reconstruction_error": round(float(np.mean(recon_errors)), 4),
        "graph_node_degree_max": int(df_graph["graph_node_degree"].max()),
        "gnn_mean_risk": round(float(np.mean(gnn_risks)), 4),
        "stream_events_processed": len(stream_res),
        "automl_result": automl_res,
        "rl_adjusted_threshold": round(new_thresh, 4),
        "online_update_prediction": round(online_prob, 4),
    }

    out_file = Path("reports/advanced_ai/advanced_ai_summary.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(summary_report, f, indent=2)

    logger.info(f"Part 17 Advanced AI & Modeling Pipeline completed successfully. Report saved to {out_file}")


if __name__ == "__main__":
    main()
