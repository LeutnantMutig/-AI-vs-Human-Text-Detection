import numpy as np
import joblib
import os

from scipy import sparse

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report, roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


# ===============================
# LOAD FEATURES
# ===============================

X_train = sparse.load_npz("features/saved/X_train.npz")
X_test = sparse.load_npz("features/saved/X_test.npz")

y_train = np.load("features/saved/y_train.npy")
y_test = np.load("features/saved/y_test.npy")

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


# ===============================
# 🔥 MODEL 1: LOGISTIC REGRESSION (BEST BASELINE)
# ===============================

print("\n🚀 Training Logistic Regression...")

lr_model = LogisticRegression(
    max_iter=3000,
    C=2,
    class_weight="balanced"
)

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)
lr_prob = lr_model.predict_proba(X_test)[:, 1]


# ===============================
# 🔥 MODEL 2: XGBOOST (GPU SAFE VERSION)
# ===============================

print("\n🚀 Training XGBoost (GPU SAFE)...")

scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

xgb_model = XGBClassifier(
    n_estimators=200,        # 🔥 reduced (prevents OOM)
    max_depth=6,             # 🔥 reduced
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    tree_method="hist",
    device="cuda",
    scale_pos_weight=scale_pos_weight,
    reg_lambda=1,
    random_state=42
)

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)
xgb_prob = xgb_model.predict_proba(X_test)[:, 1]


# ===============================
# EVALUATION FUNCTION
# ===============================

def evaluate_model(name, y_true, y_pred, y_prob):
    print(f"\n========== {name} ==========")

    print("Accuracy :", round(accuracy_score(y_true, y_pred), 4))
    print("Precision:", round(precision_score(y_true, y_pred), 4))
    print("Recall   :", round(recall_score(y_true, y_pred), 4))
    print("F1 Score :", round(f1_score(y_true, y_pred), 4))
    print("ROC-AUC  :", round(roc_auc_score(y_true, y_prob), 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))


# ===============================
# RESULTS
# ===============================

evaluate_model("Logistic Regression (BEST)", y_test, lr_pred, lr_prob)
evaluate_model("XGBoost (GPU)", y_test, xgb_pred, xgb_prob)


# ===============================
# SAVE BEST MODEL
# ===============================

os.makedirs("models/saved", exist_ok=True)

lr_acc = accuracy_score(y_test, lr_pred)
xgb_acc = accuracy_score(y_test, xgb_pred)

if lr_acc >= xgb_acc:
    best_model = lr_model
    best_name = "Logistic_Regression"
else:
    best_model = xgb_model
    best_name = "XGBoost"

joblib.dump(best_model, f"models/saved/{best_name}.pkl")

print(f"\n✅ Best model saved as: {best_name}.pkl")