from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _dense_if_needed(matrix):
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return matrix


def evaluate_model(model, preprocessor, x_test, y_test, threshold: float = 0.5) -> Dict[str, object]:
    x_test_processed = _dense_if_needed(preprocessor.transform(x_test))
    y_prob = model.predict_proba(x_test_processed)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    try:
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = 0.0

    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=["Không nguy cơ", "Nguy cơ"],
        output_dict=True,
        zero_division=0,
    )

    return {
        "y_prob": y_prob,
        "y_pred": y_pred,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_pos": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        "recall_pos": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "f1_pos": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "auc": auc,
        "cm": cm,
        "fpr": fpr,
        "tpr": tpr,
        "report_dict": report_dict,
    }


def format_prediction(prob: float, threshold: float = 0.5):
    pred = int(prob >= threshold)
    pred_label = "YES" if pred == 1 else "NO"
    display_label = "Bệnh nhân có nguy cơ tái nhập viện sớm" if pred == 1 else "Bệnh nhân không có nguy cơ tái nhập viện sớm"
    return pred, pred_label, display_label
