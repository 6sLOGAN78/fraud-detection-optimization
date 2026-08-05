# Candidate Models & Advanced AI Framework (`src/models/`)

The `src/models/` package implements candidate classifiers, stacking ensembles, graph analytics, deep learning models, and reinforcement learning policy agents.

---

## 🤖 Model Hierarchy & Architecture

```
                               ┌───────────────────────────┐
                               │   Candidate Model Suite   │
                               └─────────────┬─────────────┘
                                             │
             ┌──────────────────┬────────────┼────────────┬──────────────────┐
             ▼                  ▼            ▼            ▼                  ▼
       [LightGBM Engine]   [XGBoost]    [CatBoost]   [Deep Residual MLP] [Tabular Transformer]
             │                  │            │            │                  │
             └──────────────────┴─────┬──────┴────────────┴──────────────────┘
                                      │
                                      ▼
                        [Stacking Meta-Learner (Logistic)]
                                      │
                                      ▼
                        [GNN Risk & RL Dynamic Threshold]
```

---

## 🚀 Supported Model Paradigms

1. **Gradient Boosted Trees**: LightGBM, XGBoost, CatBoost.
2. **Stacking Ensembles**: Out-of-fold probability blending with logistic regression meta-learner.
3. **Graph Neural Networks (GNN)**: Transaction entity bipartite network graph construction with PageRank, node degree, and GNN message passing risk scoring.
4. **Deep Learning & Transformers**: Tabular Deep Residual MLP and Multi-Head Self-Attention Tabular Transformer.
5. **Reinforcement Learning Policy Agent**: Q-learning environment dynamically adjusting decision thresholds (`0.20 - 0.80`) based on recent chargeback rate feedback.
