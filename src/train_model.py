
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train = pd.read_parquet(DATA_DIR / "X_train.parquet")
    X_val = pd.read_parquet(DATA_DIR / "X_val.parquet")
    X_test = pd.read_parquet(DATA_DIR / "X_test.parquet")

    y_train = pd.read_parquet(DATA_DIR / "y_train.parquet").iloc[:, 0]
    y_val = pd.read_parquet(DATA_DIR / "y_val.parquet").iloc[:, 0]
    y_test = pd.read_parquet(DATA_DIR / "y_test.parquet").iloc[:, 0]

    # Give fraud cases greater weight because they are much rarer.
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    print("Training XGBoost...")
    model.fit(X_train, y_train, verbose=False)

    # Select a decision threshold using validation data only.
    val_probabilities = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(
        y_val, val_probabilities
    )

    # The final precision/recall point has no corresponding threshold.
    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )

    best_index = int(np.argmax(f1_scores))
    threshold = float(thresholds[best_index])

    # Evaluate on the held-out test set.
    test_probabilities = model.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= threshold).astype(int)

    average_precision = average_precision_score(
        y_test, test_probabilities
    )
    roc_auc = roc_auc_score(y_test, test_probabilities)

    report = classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        target_names=["Legitimate", "Fraud"],
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test, test_predictions, labels=[0, 1]
    )

    joblib.dump(model, MODEL_DIR / "xgboost_model.pkl")
    joblib.dump(threshold, MODEL_DIR / "threshold.pkl")
    joblib.dump(list(X_train.columns), MODEL_DIR / "feature_columns.pkl")

    metrics = {
        "threshold": threshold,
        "average_precision": float(average_precision),
        "roc_auc": float(roc_auc),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "scale_pos_weight": float(scale_pos_weight),
    }

    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    predictions_df = pd.DataFrame({
        "actual": y_test.to_numpy(),
        "fraud_probability": test_probabilities,
        "predicted": test_predictions,
    })
    predictions_df.to_csv(
        RESULTS_DIR / "test_predictions.csv", index=False
    )

    importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance.to_csv(
        RESULTS_DIR / "feature_importance.csv", index=False
    )

    print("\n=== Test Results ===")
    print(f"Decision threshold: {threshold:.4f}")
    print(f"Average precision (PR-AUC): {average_precision:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("\nClassification report:")
    print(classification_report(
        y_test, test_predictions,
        target_names=["Legitimate", "Fraud"],
        zero_division=0,
    ))
    print("Confusion matrix:")
    print(matrix)
    print(f"\nModel saved to: {MODEL_DIR}")
    print(f"Metrics saved to: {RESULTS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
