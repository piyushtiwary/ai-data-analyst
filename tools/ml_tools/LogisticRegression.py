from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List

import joblib
import pandas as pd

from langchain.tools import tool

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from .model_utils import (
    build_preprocessor,
    build_smote_pipeline,
    split_dataset,
    undersample_training_data,
    compute_metrics,
    normalize_feature_importance,
    compute_cv_scores,
)


@tool
def train_logistic_regression_tool(
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
    Train and evaluate a Logistic Regression classifier.

    Handles optional class imbalance strategies and stores the trained pipeline.
    """
    df = pd.read_csv(dataset_path)

    X_train, X_test, y_train, y_test = split_dataset(
        df,
        target_column,
        test_size,
        random_state,
    )

    applied_strategy = imbalance_strategy
    if imbalance_strategy == "undersample":
        X_train, y_train = undersample_training_data(X_train, y_train, random_state)

    use_smote = imbalance_strategy == "smote"

    preprocessor = build_preprocessor(
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        ordinal_columns=ordinal_columns,
        scaling_required=scaling_required,
        dense_output=use_smote,
    )

    model_kwargs: Dict[str, Any] = {
        "max_iter": 1000,
        "solver": "lbfgs",
    }

    if imbalance_strategy == "class_weight":
        model_kwargs["class_weight"] = "balanced"

    model = LogisticRegression(**model_kwargs)

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

        sample_weights = None
        if imbalance_strategy == "class_weight":
            sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

        pipeline.fit(X_train, y_train, model__sample_weight=sample_weights)

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

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    coefficients = pipeline.named_steps["model"].coef_[0]

    feature_importance = normalize_feature_importance(
        feature_names.tolist(),
        [abs(value) for value in coefficients.tolist()],
    )

    os.makedirs("outputs/models", exist_ok=True)
    experiment_id = str(uuid.uuid4())[:8]
    model_path = f"outputs/models/logreg_{experiment_id}.pkl"
    joblib.dump(pipeline, model_path)

    return {
        "experiment_id": experiment_id,
        "model_name": "LogisticRegression",
        "model_path": model_path,
        "metrics": metrics,
        "feature_importance": feature_importance,
        **cv_stats,
        "hyperparameters": {
            "max_iter": 1000,
            "solver": "lbfgs",
            "class_weight": model_kwargs.get("class_weight", "none"),
            "imbalance_strategy": applied_strategy,
        },
    }
