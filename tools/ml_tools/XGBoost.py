# tools/xgboost_tool.py

from langchain.tools import tool

from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import pandas as pd

import joblib

import os


@tool
def train_xgboost_classifier_tool(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train,
    y_test,
    categorical_columns: list,
    numerical_columns: list,
    ordinal_columns: list = [],
):
    """
    Train and evaluate an XGBoost classifier.
    Saves trained model locally.
    Returns metrics and model path.
    """

    # Build Encoders
    transformers = []

    one_hot_columns = [col for col in categorical_columns if col not in ordinal_columns]

    # One-hot categorical encoding
    if one_hot_columns:

        transformers.append(
            ("onehot", OneHotEncoder(handle_unknown="ignore"), one_hot_columns)
        )

    # Ordinal encoding
    if ordinal_columns:

        transformers.append(("ordinal", OrdinalEncoder(), ordinal_columns))

    # Numeric scaling
    if numerical_columns:

        transformers.append(("num", StandardScaler(), numerical_columns))

    # Preprocessor
    preprocessor = ColumnTransformer(transformers=transformers)

    # XGBoost Model
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )

    # Full Pipeline
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

    # Train
    pipeline.fit(X_train, y_train)

    # Predictions
    predictions = pipeline.predict(X_test)

    # Metrics
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions)),
        "recall": float(recall_score(y_test, predictions)),
        "f1_score": float(f1_score(y_test, predictions)),
    }

    # Feature Importance
    booster = pipeline.named_steps["model"]

    feature_importance = dict(
        zip(numerical_columns, booster.feature_importances_[: len(numerical_columns)])
    )

    # Save Model
    os.makedirs("artifacts/models", exist_ok=True)

    model_path = "artifacts/models/" "xgboost_classifier.pkl"

    joblib.dump(pipeline, model_path)

    # Return Structured Result
    return {
        "model_name": "XGBoostClassifier",
        "model_path": model_path,
        "metrics": metrics,
        "feature_importance": feature_importance,
    }
