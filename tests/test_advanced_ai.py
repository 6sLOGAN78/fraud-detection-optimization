"""Unit tests for Part 17 — Advanced Fraud Modeling, Graph & Deep Learning, Streaming & Federated AI."""

import numpy as np
import pandas as pd
import pytest
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


@pytest.fixture
def sample_ai_data():
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(20, 4), columns=["f1", "f2", "f3", "f4"])
    y = np.random.randint(0, 2, 20)
    return X, y


def test_17_1_to_17_3_ensembles(sample_ai_data):
    X, y = sample_ai_data

    b1 = RandomForestClassifier(n_estimators=5, random_state=42)
    b2 = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=43)

    stacker = AdvancedStackingMetaLearner(base_models=[b1, b2], n_splits=2)
    stacker.fit(X, y)
    probs = stacker.predict_proba(X)
    assert probs.shape == (20, 2)

    pseudo = SemiSupervisedPseudoLabeler(high_confidence_threshold=0.7, low_confidence_threshold=0.3)
    X_p, y_p = pseudo.generate_pseudo_labels(b1, X)
    assert len(X_p) == len(y_p)

    auto = SelfSupervisedAutoencoder(input_dim=4, hidden_dim=2)
    auto.fit(X)
    errors = auto.compute_reconstruction_error(X)
    assert len(errors) == 20


def test_17_4_and_17_7_graph(sample_ai_data):
    X, y = sample_ai_data

    graph_eng = TransactionGraphEngine()
    df_g = graph_eng.compute_graph_metrics(X)
    assert "graph_node_degree" in df_g.columns
    assert "graph_pagerank" in df_g.columns

    gnn = GNNFraudDetector(input_dim=4)
    gnn.fit(X, y)
    risks = gnn.predict_node_risk(X)
    assert len(risks) == 20


def test_17_5_and_17_6_deep_learning(sample_ai_data):
    X, y = sample_ai_data

    mlp = TabularDeepMLP(input_dim=4)
    mlp.fit(X, y)
    probs_mlp = mlp.predict_proba(X)
    assert probs_mlp.shape == (20, 2)

    transformer = TabularTransformerModel(input_dim=4)
    transformer.fit(X, y)
    probs_tr = transformer.predict_proba(X)
    assert probs_tr.shape == (20, 2)


def test_17_8_to_17_12_advanced_ai(sample_ai_data):
    X, y = sample_ai_data

    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)

    stream = StreamingInferenceEngine(model=model, feature_names=["f1", "f2", "f3", "f4"])
    events = [X.iloc[0].to_dict(), X.iloc[1].to_dict()]
    res = stream.process_event_stream(events)
    assert len(res) == 2

    fed = FederatedLearningAggregator()
    w1 = np.ones(4)
    w2 = np.ones(4) * 3
    avg_w = fed.aggregate_client_weights([w1, w2])
    assert np.allclose(avg_w, 2.0)

    automl = AutoMLEngine()
    res_am = automl.run_automl_search(X, y)
    assert "best_algorithm" in res_am

    rl = FraudRLPolicyAgent(initial_threshold=0.5)
    t = rl.select_action_threshold(0.08)
    assert t < 0.5

    online = ContinuousOnlineLearner(feature_names=["f1", "f2", "f3", "f4"])
    prob = online.update_online(X.iloc[0].to_dict(), y_true=1)
    assert 0.0 <= prob <= 1.0
