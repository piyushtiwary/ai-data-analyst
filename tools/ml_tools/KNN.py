from __future__ import annotations

import os
import uuid
from typing import Any, Dict

import joblib
import pandas as pd

from langchain.tools import tool

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from .model_utils import (
    build_preprocessor,
    build_smote_pipeline,
    split_dataset,
    undersample_training_data,
    compute_metrics,
    compute_cv_scores,
)


@tool
def train_knn_tool(
    dataset_path: str,
    target_column: str,
    categorical_columns: list[str],
    numerical_columns: list[str],
    ordinal_columns: list[str] = [],
    scaling_required: bool = True,
    imbalance_strategy: str = "none",
    cv_folds: int = 5,
    cv_repeats: int = 2,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train and evaluate a K-Nearest Neighbors classifier.
    """
    df = pd.read_csv(dataset_path)

    X_train, X_test, y_train, y_test = split_dataset(
        df,
        target_column,
        test_size,
        random_state,
    )

    applied_strategy = imbalance_strategy
    if imbalance_strategy in {"undersample", "class_weight"}:
        X_train, y_train = undersample_training_data(X_train, y_train, random_state)
        applied_strategy = "undersample"

    use_smote = imbalance_strategy == "smote"

    preprocessor = build_preprocessor(
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        ordinal_columns=ordinal_columns,
        scaling_required=scaling_required,
        dense_output=use_smote,
    )

    model = KNeighborsClassifier(n_neighbors=7, weights="distance")

    if use_smote:
        pipeline = build_smote_pipeline(preprocessor, model, random_state)
        pipeline.fit(X_train, y_train)
    else:
        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)

    cv_stats = compute_cv_scores(
        pipeline,
        X_train,
        y_train,
        cv_folds=cv_folds,
        cv_repeats=cv_repeats,
        scoring="roc_auc",
    )

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, predictions, probabilities)

    os.makedirs("outputs/models", exist_ok=True)
    experiment_id = str(uuid.uuid4())[:8]
    model_path = f"outputs/models/knn_{experiment_id}.pkl"
    joblib.dump(pipeline, model_path)

    return {
        "experiment_id": experiment_id,
        "model_name": "KNeighborsClassifier",
        "model_path": model_path,
        "metrics": metrics,
        "feature_importance": {},
        **cv_stats,
        "hyperparameters": {
            "n_neighbors": 7,
            "weights": "distance",
            "imbalance_strategy": applied_strategy,
        },
    }
