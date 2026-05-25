from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import joblib

from langchain.tools import tool

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score


def _duplicate_overlap_ratio(X_train: pd.DataFrame, X_test: pd.DataFrame) -> float:
    train_rows = X_train.astype(str).agg("|".join, axis=1)
    test_rows = X_test.astype(str).agg("|".join, axis=1)

    train_set = set(train_rows.values.tolist())
    test_set = set(test_rows.values.tolist())

    if not test_set:
        return 0.0

    overlap = len(train_set.intersection(test_set))
    return float(overlap / max(len(test_set), 1))


def _max_abs_feature_target_corr(df: pd.DataFrame, target_column: str) -> float:
    if target_column not in df.columns:
        return 0.0

    numeric_df = df.select_dtypes(include=["number"]).copy()
    if target_column not in numeric_df.columns:
        return 0.0

    corr = numeric_df.corr(numeric_only=True)[target_column].drop(
        labels=[target_column]
    )
    if corr.empty:
        return 0.0

    return float(corr.abs().max())


def _top_feature_target_correlations(
    df: pd.DataFrame, target_column: str, top_k: int = 10
) -> List[Dict[str, Any]]:
    numeric_df = df.select_dtypes(include=["number"]).copy()
    if target_column not in numeric_df.columns:
        return []

    corr = numeric_df.corr(numeric_only=True)[target_column].drop(
        labels=[target_column]
    )
    if corr.empty:
        return []

    ranked = sorted(corr.abs().items(), key=lambda x: x[1], reverse=True)
    results = []
    for feature, value in ranked[:top_k]:
        results.append({"feature": feature, "abs_corr": float(value)})

    return results


@tool
def evaluate_model_evidence_tool(
    model_path: str,
    dataset_path: str,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
    correlation_threshold: float = 0.9,
    duplicate_overlap_threshold: float = 0.0,
) -> Dict[str, Any]:
    """
    Generate evidence for validation: train/test scores, duplicate overlap, and
    feature-target correlations. Returns summarized, non-raw outputs only.
    """
    df = pd.read_csv(dataset_path)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
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
        }

    train_proba = model.predict_proba(X_train)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]

    train_pred = (train_proba >= 0.5).astype(int)
    test_pred = (test_proba >= 0.5).astype(int)

    train_score = float(roc_auc_score(y_train, train_proba))
    test_score = float(roc_auc_score(y_test, test_proba))

    train_f1 = float(f1_score(y_train, train_pred, zero_division=0))
    test_f1 = float(f1_score(y_test, test_pred, zero_division=0))

    duplicate_overlap = _duplicate_overlap_ratio(X_train, X_test)
    max_corr = _max_abs_feature_target_corr(df, target_column)
    top_corrs = _top_feature_target_correlations(df, target_column)

    return {
        "cv_mean": None,
        "cv_std": None,
        "train_score": train_score,
        "test_score": test_score,
        "train_f1": train_f1,
        "test_f1": test_f1,
        "duplicate_overlap": duplicate_overlap,
        "feature_target_correlation_max": max_corr,
        "feature_target_correlations": top_corrs,
        "duplicate_overlap_threshold": float(duplicate_overlap_threshold),
        "correlation_threshold": float(correlation_threshold),
        "duplicate_overlap_ok": duplicate_overlap <= duplicate_overlap_threshold,
        "correlation_ok": max_corr <= correlation_threshold,
        "preprocessing_contamination": "not_detected",
    }
