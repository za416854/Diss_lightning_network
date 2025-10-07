import json
import numpy as np # fastly do array calculation
import random
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
# load features and labels 
with open("node_feature_vectors.json") as f:
    node_features = json.load(f)

with open("bad_nodes.json") as f:
    bad_nodes = set(json.load(f))

test_size = 0.2

# dynamic features I have
dynamic_feature_flags = {
    "as_source": True,
    "as_target": True,
    "forwarding_events_count": True,
    "success_rate": True,
    "jamming_attempts": True,
    "jamming_ratio": True,
    "avg_slot_utilization": True,
    "degree_centrality": True, 
    "betweenness_centrality": True, 
    "clustering": True,
    "avg_base_fee_msat": True,
    "avg_fee_rate_milli_msat": True,
    "avg_max_htlc_msat": True,
    "capacity_centrality": True, 
}

# collect all static feature_* keys that appear in the JSON
all_keys = set()
for feats in node_features.values():
    all_keys.update(feats.keys())

# keep only feature_* keys
static_feature_names = [k for k in all_keys if k.startswith("feature_")]

# make the order stable (sort by numeric part if possible)
def feature_key(k):
    try:
        return int(k.split("_")[1])
    except Exception:
        return 10**9  # put weird keys at the end
    
# sort static_feature_names by the value returned by feature_key
static_feature_names = sorted(static_feature_names, key=feature_key) 

# final feature name list (dynamic first, then static)
dynamic_feature_names = [k for k, v in dynamic_feature_flags.items() if v]
feature_names = dynamic_feature_names + static_feature_names

print(" Using features:")
for n in feature_names:
    print("  -", n)


# load node_feature_vectors.json and convert to dataFrame by Pandas
df = pd.read_json("node_feature_vectors.json", orient='index')

# reorder and filter the columns based on the feature_names list to ensure the correct order and convert the bool values ​​to 0/1
X = df[feature_names].astype(float)

if X.isnull().values.any():
    print("\n WARNING: NaN missing values ​​found in data")
    nan_cols = X.columns[X.isnull().any()].tolist()
    print(f" features that generate NaN: {nan_cols}")
else:
    print("data check completed, no NaN values ​​found。")

# create labels y (Series) and node_ids (list).
node_ids = df.index.tolist()
y = pd.Series([1 if node_id in bad_nodes else 0 for node_id in node_ids], index=node_ids)

# convert y to a NumPy array to meet the requirements of scikit-learn functions
y = y.to_numpy()

# convert X to a NumPy array to meet the requirements of scikit-learn functions
X_df = X # keep a DataFrame version of X for SHAP
X = X.to_numpy()
print(f"\n Total nodes: {len(X)}")

# ensure at least one bad node in test
bad_indices = [i for i, nid in enumerate(node_ids) if nid in bad_nodes]
if not bad_indices:
    raise ValueError("No bad nodes found in labels (bad_nodes.json).")
test_bad_index = random.choice(bad_indices)
remaining_indices = list(set(range(len(X))) - {test_bad_index}) # subtact that index means ensure at least one node in the test
random.shuffle(remaining_indices)

test_size_count = int(len(X) * test_size) - 1
test_indices = [test_bad_index] + remaining_indices[:max(test_size_count, 0)]
train_indices = list(set(range(len(X))) - set(test_indices))

X_train = X[train_indices]
y_train = y[train_indices]
id_train = [node_ids[i] for i in train_indices]

X_test = X[test_indices]
y_test = y[test_indices]
id_test = [node_ids[i] for i in test_indices]

print(f" Training size: {len(X_train)} | Testing size: {len(X_test)}")
print(" Bad nodes in test:",
      [nid[:6] for nid in id_test if nid in bad_nodes])

#  Model selection options: logistic, random_forest, svm, xgboost
MODEL_TYPE = "xgboost"  

print(f"\n MODEL_TYPE  is {MODEL_TYPE}")

if MODEL_TYPE == "logistic":
    clf = LogisticRegression(max_iter=5000) # class_weight="balanced" is not added here since LR is highly sensitive to this parameter
elif MODEL_TYPE == "random_forest":
    clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
elif MODEL_TYPE == "svm":
    clf = SVC(probability=True, kernel="rbf", class_weight="balanced")

elif MODEL_TYPE == "xgboost":
    clf = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        learning_rate=0.05,            
        gamma=0.2,                     
        max_depth=100,                
        colsample_bylevel=0.75,       
        subsample=0.75,               
        n_estimators=200,             
        random_state=42
    )
else:
    raise ValueError("Unsupported MODEL_TYPE")


from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score


# use 60/20/20 split：Train / Validation / Test
print("\n Final Model Training on 60% | Validation 20% | Test 20%")

X_temp, X_test_final, y_temp, y_test_final, node_temp, node_test = train_test_split(
    X, y, node_ids, test_size=0.2, stratify=y, random_state=42
)

X_train_final, X_val_final, y_train_final, y_val_final, node_train, node_val = train_test_split(
    X_temp, y_temp, node_temp, test_size=0.25, stratify=y_temp, random_state=42
)  # test_size = 0.8 x 0.25 = 0.2 for final testing

print(f" Final Split Sizes → Train: {len(X_train_final)}, Val: {len(X_val_final)}, Test: {len(X_test_final)}")
# display the first 6 digits of the bad node pub_key
def short(nid):
    return nid[:6]

def analyze_partition(name, ids):
    bad = [nid for nid in ids if nid in bad_nodes]
    good = [nid for nid in ids if nid not in bad_nodes]
    short_ids = [short(nid) for nid in bad]
    print(f"\n {name} Partition")
    print(f" - Total Nodes: {len(ids)}")
    print(f" - Good: {len(good)} | Bad: {len(bad)}")
    print(f" - Bad Node pub_keys: {short_ids}")

analyze_partition("Train", node_train)
analyze_partition("Validation", node_val)
analyze_partition("Test", node_test)


if MODEL_TYPE == "logistic":
    final_model = LogisticRegression(max_iter=5000) # class_weight="balanced" is not added here since LR is highly sensitive to this parameter
elif MODEL_TYPE == "random_forest":
    final_model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
elif MODEL_TYPE == "svm":
    final_model = SVC(probability=True, kernel="rbf", class_weight="balanced")
elif MODEL_TYPE == "xgboost":
    final_model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=(y_train_final == 0).sum() / (y_train_final == 1).sum(),
        learning_rate=0.05,
        gamma=0.2,
        max_depth=100,
        colsample_bylevel=0.75,
        subsample=0.75,
        n_estimators=200,
        random_state=42
    )


else:
    raise ValueError("Unsupported MODEL_TYPE")
final_model.fit(X_train_final, y_train_final)
# because y is the label, that means the correct answer, it is not included in the test, it is used to check the answer
y_probs = final_model.predict_proba(X_test_final)[:, 1] 
y_preds = (y_probs >= 0.4).astype(int)

# Predicted Train and Val Scores (Report)
print("\n Train Set Classification Report:")
y_train_pred = final_model.predict(X_train_final)
print(classification_report(y_train_final, y_train_pred))

print("\n Validation Set Classification Report:")
y_val_pred = final_model.predict(X_val_final)
print(classification_report(y_val_final, y_val_pred))

print("\n Classification Report:")
print(classification_report(y_test_final, y_preds))
print(" Confusion Matrix:")
print(confusion_matrix(y_test_final, y_preds))

# feature importance
if MODEL_TYPE == "logistic":
    print("\n Logistic Coefficients (sign = direction):")
    for name, coef in sorted(zip(feature_names, final_model.coef_[0]), key=lambda x: abs(x[1]), reverse=True):
        direction = '+' if coef > 0 else '-'
        print(f"{name:25}: {direction}{abs(coef):.6f}")
if hasattr(final_model, "feature_importances_"):
    print("\n Feature Importances:")
    for name, importance in sorted(zip(feature_names, final_model.feature_importances_), key=lambda x: -x[1]):
        print(f"{name:25}: {importance:.4f}")

# SHAP digital output
import shap
import matplotlib.pyplot as plt
print("\n\n-- Starting SHAP Analysis --")
try:
    # in order to let SHAP to know the feature names, need to convert the NumPy back to a DataFrame
    X_train_df = pd.DataFrame(X_train_final, columns=X_df.columns)
    X_test_df = pd.DataFrame(X_test_final, columns=X_df.columns)

    # build up shap.Explainer(pass in a DataFrame with column names)
    explainer = shap.Explainer(final_model, X_train_df)

    # get SHAP values for the test set(for the test set)
    shap_values = explainer(X_test_df)

    # only get class 1 SHAP value only get class 1 SHAP value (bad nodes)
    if len(shap_values.shape) == 3:
        shap_values_class_1 = shap_values[:, :, 1]
    else:
        shap_values_class_1 = shap_values

    # calculate and display global feature importances (average of absolute values)
    mean_abs_shap = np.abs(shap_values_class_1.values).mean(axis=0)

    # create a clean DataFrame for display
    shap_importance_df = pd.DataFrame({
        'feature': shap_values_class_1.feature_names,
        'importance': mean_abs_shap
    }).sort_values('importance', ascending=False)

    print("\n SHAP Global Feature Importances (mean absolute value):")
    print(shap_importance_df.to_string(index=False))
    shap.summary_plot(shap_values, X_test_df, show=False, max_display=3)
    plt.tight_layout()   
    plt.savefig("shap_summary.png", bbox_inches="tight") 
    plt.close()


except Exception as e:
    print(f" SHAP numeric analysis skipped: {e}")
# sort Suspicious nodes 
print("\n Suspicious Nodes (by probability):")
sorted_nodes = sorted(zip(node_test, y_probs), key=lambda x: -x[1])

# use threshold (0.4) to judge bad nodes
threshold = 0.4
predicted_bad_nodes = {nid for nid, prob in zip(node_test, y_probs) if prob >= threshold}
tp = fp = fn = tn = 0
for nid, prob in sorted_nodes:
    is_bad = nid in bad_nodes
    predicted = nid in predicted_bad_nodes
    label = "❌ Bad" if predicted else "✅ Good"
    prefix = "⭐" if is_bad else ""
    print(f"{prefix} Node {nid[:6]}...: {label} (Prob = {prob:.2f})")

    if predicted and is_bad:
        tp += 1
    elif predicted and not is_bad:
        fp += 1
    elif not predicted and is_bad:
        fn += 1
    else:
        tn += 1

precision = tp / (tp + fp) if (tp + fp) else 0
recall = tp / (tp + fn) if (tp + fn) else 0
print(f"\n Final Precision: {precision:.2f}")
print(f" Final Recall: {recall:.2f}")

