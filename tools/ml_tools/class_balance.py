from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from langchain.tools import tool


@tool
def analyze_class_balance_tool(dataset_path: str, target_column: str) -> Dict[str, Any]:
    """
    Analyze class balance for a target column.

    Returns class distribution and imbalance indicators only.
    """
    df = pd.read_csv(dataset_path)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    value_counts = df[target_column].value_counts(dropna=False)
    total = int(value_counts.sum())

    if total == 0:
        return {
            "class_distribution": {},
            "total_samples": 0,
            "minority_class": None,
            "majority_class": None,
            "minority_rate": 0.0,
            "imbalance_ratio": 0.0,
            "is_imbalanced": False,
        }

    majority_class = value_counts.idxmax()
    minority_class = value_counts.idxmin()

    minority_count = int(value_counts.min())
    majority_count = int(value_counts.max())

    minority_rate = minority_count / total
    imbalance_ratio = majority_count / max(minority_count, 1)

    return {
        "class_distribution": value_counts.to_dict(),
        "total_samples": total,
        "minority_class": minority_class,
        "majority_class": majority_class,
        "minority_rate": float(minority_rate),
        "imbalance_ratio": float(imbalance_ratio),
        "is_imbalanced": bool(minority_rate < 0.2 or imbalance_ratio >= 3.0),
    }
