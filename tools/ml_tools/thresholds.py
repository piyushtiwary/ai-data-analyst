from __future__ import annotations

from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

from langchain.tools import tool

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score


@tool
def optimize_thresholds_tool(
    model_path: str,
    dataset_path: str,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
    min_precision: float = 0.8,
) -> Dict[str, Any]:
    """
    Optimize classification thresholds using a fresh stratified split.
    Returns best thresholds for F1 and for precision-recall tradeoff.
    """
    df = pd.read_csv(dataset_path)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_eval, y_train, y_eval = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = joblib.load(model_path)

    if not hasattr(model, "predict_proba"):
        return {
            "error": "Model does not support probability predictions.",
            "threshold_scores": [],
        }

    proba = model.predict_proba(X_eval)[:, 1]

    thresholds = np.linspace(0.05, 0.95, 19)

    best_f1 = -1.0
    best_f1_threshold = 0.5
    best_f1_precision = 0.0
    best_f1_recall = 0.0

    best_recall = -1.0
    best_recall_threshold = 0.5
    best_recall_precision = 0.0

    fold_scores: List[Dict[str, float]] = []

    for threshold in thresholds:
        preds = (proba >= threshold).astype(int)

        precision = precision_score(y_eval, preds, zero_division=0)
        recall = recall_score(y_eval, preds, zero_division=0)
        f1 = f1_score(y_eval, preds, zero_division=0)

        fold_scores.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )

        if f1 > best_f1:
            best_f1 = float(f1)
            best_f1_threshold = float(threshold)
            best_f1_precision = float(precision)
            best_f1_recall = float(recall)

        if precision >= min_precision and recall > best_recall:
            best_recall = float(recall)
            best_recall_threshold = float(threshold)
            best_recall_precision = float(precision)

    return {
        "best_f1_threshold": best_f1_threshold,
        "best_f1": best_f1,
        "best_f1_precision": best_f1_precision,
        "best_f1_recall": best_f1_recall,
        "best_recall_threshold": best_recall_threshold,
        "best_recall": best_recall,
        "best_recall_precision": best_recall_precision,
        "min_precision": float(min_precision),
        "threshold_scores": fold_scores,
    }
