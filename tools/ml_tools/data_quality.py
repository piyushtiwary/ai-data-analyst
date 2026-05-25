from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from langchain.tools import tool


@tool
def assess_data_quality_tool(
    dataset_path: str,
    target_column: str,
) -> Dict[str, Any]:
    """
    Summarize dataset quality signals (missingness, duplicates, diversity) without raw data.
    """
    df = pd.read_csv(dataset_path)

    total_rows = int(len(df))
    total_columns = int(len(df.columns))

    missing_rate = float(df.isnull().mean().mean()) if total_rows else 0.0
    duplicate_rows = int(df.duplicated().sum())
    duplicate_rate = float(duplicate_rows / max(total_rows, 1))

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    cardinality = {col: int(df[col].nunique(dropna=False)) for col in categorical_cols}

    low_variance_numeric = []
    for col in numeric_cols:
        if df[col].std() == 0:
            low_variance_numeric.append(col)

    target_cardinality = (
        int(df[target_column].nunique()) if target_column in df.columns else 0
    )

    return {
        "rows": total_rows,
        "columns": total_columns,
        "missing_rate": missing_rate,
        "duplicate_rate": duplicate_rate,
        "categorical_cardinality": cardinality,
        "low_variance_numeric": low_variance_numeric,
        "target_cardinality": target_cardinality,
        "limitations": [
            (
                "No temporal fields detected"
                if "date" not in " ".join(df.columns).lower()
                else "Temporal features present"
            ),
            (
                "Potential demographic coverage unknown"
                if "gender" not in [c.lower() for c in df.columns]
                else "Demographic feature present"
            ),
        ],
    }
