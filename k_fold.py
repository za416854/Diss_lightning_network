import json, numpy as np, random
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, precision_score, recall_score


with open("node_feature_vectors.json") as f: feats = json.load(f)
with open("bad_nodes.json") as f: bad_nodes = set(json.load(f))

feature_names = [k for k in next(iter(feats.values())) if k.startswith("feature_")]

feature_names += [
    "as_source",
    "as_target",
    "forwarding_events_count", 
    "success_rate",
    "degree_centrality",
    "clustering",
    "avg_base_fee_msat",
    "avg_fee_rate_milli_msat",
    "avg_max_htlc_msat",
    "capacity_centrality",
    "betweenness_centrality",
    "jamming_attempts",
    "jamming_ratio",
    "avg_slot_utilization"
]
X, y = [], []
for nid, vec in feats.items():
    row = [vec.get(k, 0) for k in feature_names]
    X.append(row)
    y.append(1 if nid in bad_nodes else 0)
X, y = np.array(X), np.array(y)

model_scores = []

# K-Fold validation to compare models
model_scores = []

for model_type in ["logistic", "random_forest", "svm", "xgboost"]:
    print(f"\n Testing model: {model_type}")
    precision_scores, recall_scores, f1_scores = [], [], []

    def get_model():
        if model_type == "logistic": return LogisticRegression(max_iter=5000) # class_weight="balanced" is not added here since LR is highly sensitive to this parameter
        if model_type == "random_forest": return RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
        if model_type == "svm": return SVC(probability=True, kernel="rbf", class_weight="balanced")
        
        raise ValueError("Unknown model_type")

    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for i, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
        if model_type == "xgboost":
            neg, pos = np.bincount(y[train_idx])
            scale_pos_weight = neg / max(pos, 1)
            model = XGBClassifier(
                use_label_encoder=False,
                eval_metric='logloss',
                scale_pos_weight=scale_pos_weight,
                learning_rate=0.05,
                gamma=0.2,
                max_depth=100,
                colsample_bylevel=0.75,
                subsample=0.75,
                n_estimators=200,
                random_state=42
            )
        else:
            model = get_model()
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[val_idx])
        print(f"\n Fold {i} Report:")
        print(classification_report(y[val_idx], y_pred, zero_division=0))
        p = precision_score(y[val_idx], y_pred, zero_division=0)
        r = recall_score(y[val_idx], y_pred, zero_division=0)
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        print(f" Precision: {p:.3f},  Recall: {r:.3f},  F1: {f1:.3f}")
        precision_scores.append(p)
        recall_scores.append(r)
        f1_scores.append(f1)

    model_scores.append({
        "model": model_type,
        "precision": np.mean(precision_scores),
        "recall": np.mean(recall_scores),
        "f1": np.mean(f1_scores)
    })

# sort out the best models
best_model = sorted(model_scores, key=lambda x: x["f1"], reverse=True)[0]
print(f"\n Best model by F1-score: {best_model['model']} with F1 = {best_model['f1']:.4f}")
