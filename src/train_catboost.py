import json
from pathlib import Path

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

PROC = Path("data/processed")
Path("models").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)


def load(name):
    return pd.read_parquet(PROC / f"{name}.parquet")


X_train, X_val, X_test = load("X_train"), load("X_val"), load("X_test")
y_train = load("y_train").iloc[:, 0]
y_val = load("y_val").iloc[:, 0]
y_test = load("y_test").iloc[:, 0]

model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    eval_metric="PRAUC",
    auto_class_weights="Balanced",   # nécessaire : le train n'est pas rééquilibré
    early_stopping_rounds=100,
    random_seed=42,
    verbose=100,
)
model.fit(X_train, y_train, eval_set=(X_val, y_val))

# Seuil choisi sur la validation (jamais sur le test)
val_proba = model.predict_proba(X_val)[:, 1]
prec, rec, thr = precision_recall_curve(y_val, val_proba)
f1 = 2 * prec * rec / (prec + rec + 1e-12)
best_thr = float(thr[f1[:-1].argmax()])

# Évaluation finale sur le test
test_proba = model.predict_proba(X_test)[:, 1]
test_pred = (test_proba >= best_thr).astype(int)

metrics = {
    "model": "CatBoost",
    "threshold": best_thr,
    "pr_auc": float(average_precision_score(y_test, test_proba)),
    "roc_auc": float(roc_auc_score(y_test, test_proba)),
    "precision": float(precision_score(y_test, test_pred)),
    "recall": float(recall_score(y_test, test_pred)),
    "f1": float(f1_score(y_test, test_pred)),
    "confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
}
print(json.dumps(metrics, indent=2))

# Sauvegardes avec le préfixe catboost_ pour ne pas écraser les fichiers XGBoost
model.save_model("models/catboost_model.cbm")
joblib.dump(best_thr, "models/catboost_threshold.pkl")
joblib.dump(list(X_train.columns), "models/catboost_feature_columns.pkl")

Path("results/catboost_metrics.json").write_text(json.dumps(metrics, indent=2))

pd.DataFrame({
    "y_true": y_test.values,
    "proba": test_proba,
    "y_pred": test_pred,
}).to_csv("results/catboost_test_predictions.csv", index=False)

pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.get_feature_importance(),
}).sort_values("importance", ascending=False
).to_csv("results/catboost_feature_importance.csv", index=False)