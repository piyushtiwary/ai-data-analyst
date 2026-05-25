from __future__ import annotations

from typing import Any, Dict, Tuple, List

import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    RepeatedStratifiedKFold,
    cross_val_score,
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def _create_one_hot_encoder(dense_output: bool) -> OneHotEncoder:
    if dense_output:
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=False)

    return OneHotEncoder(handle_unknown="ignore")


def build_preprocessor(
    categorical_columns: list[str],
    numerical_columns: list[str],
    ordinal_columns: list[str] | None = None,
    scaling_required: bool = False,
    dense_output: bool = False,
) -> ColumnTransformer:
    ordinal_columns = ordinal_columns or []

    transformers = []

    one_hot_columns = [col for col in categorical_columns if col not in ordinal_columns]

    if one_hot_columns:
        transformers.append(
            (
                "onehot",
                _create_one_hot_encoder(dense_output),
                one_hot_columns,
            )
        )

    if ordinal_columns:
        transformers.append(
            (
                "ordinal",
                OrdinalEncoder(),
                ordinal_columns,
            )
        )

    if numerical_columns:
        if scaling_required:
            transformers.append(
                (
                    "num",
                    StandardScaler(),
                    numerical_columns,
                )
            )
        else:
            transformers.append(
                (
                    "num",
                    "passthrough",
                    numerical_columns,
                )
            )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_smote_pipeline(
    preprocessor: ColumnTransformer, model: Any, random_state: int
):
    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
    except ImportError as exc:
        raise ImportError(
            "imbalanced-learn is required for SMOTE. Add imbalanced-learn to requirements.txt."
        ) from exc

    return ImbPipeline(
        [
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=random_state)),
            ("model", model),
        ]
    )


def split_dataset(
    df: pd.DataFrame,
    target_column: str,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(columns=[target_column])
    y = df[target_column]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def undersample_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    train_df = X_train.copy()
    train_df["__target__"] = y_train.values

    class_counts = train_df["__target__"].value_counts(dropna=False)
    if class_counts.empty:
        return X_train, y_train

    min_count = int(class_counts.min())

    sampled_frames = []
    for label, group in train_df.groupby("__target__"):
        sampled_frames.append(group.sample(n=min_count, random_state=random_state))

    balanced_df = pd.concat(sampled_frames).sample(frac=1.0, random_state=random_state)

    X_balanced = balanced_df.drop(columns=["__target__"])
    y_balanced = balanced_df["__target__"]

    return X_balanced, y_balanced


def compute_metrics(
    y_true: pd.Series,
    y_pred: Any,
    y_proba: Any | None,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
    }

    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            metrics["roc_auc"] = 0.0
    else:
        metrics["roc_auc"] = 0.0

    return metrics


def normalize_feature_importance(
    feature_names: List[str], importance: List[float]
) -> Dict[str, float]:
    if not feature_names or not importance:
        return {}

    pairs = list(zip(feature_names, importance))
    pairs.sort(key=lambda x: x[1], reverse=True)

    return {name: float(score) for name, score in pairs}


def compute_cv_scores(
    pipeline: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int,
    cv_repeats: int,
    scoring: str = "roc_auc",
) -> Dict[str, Any]:
    if cv_folds < 2:
        return {
            "fold_scores": [],
            "cv_mean_roc_auc": 0.0,
            "cv_std_roc_auc": 0.0,
        }

    splitter = RepeatedStratifiedKFold(
        n_splits=cv_folds,
        n_repeats=max(cv_repeats, 1),
        random_state=42,
    )

    scores = cross_val_score(
        pipeline,
        X,
        y,
        scoring=scoring,
        cv=splitter,
        n_jobs=None,
    )

    fold_scores = [float(score) for score in scores]

    return {
        "fold_scores": fold_scores,
        "cv_mean_roc_auc": float(scores.mean()),
        "cv_std_roc_auc": float(scores.std()),
    }
