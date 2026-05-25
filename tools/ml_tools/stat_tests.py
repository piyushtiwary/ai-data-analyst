from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from langchain.tools import tool

from scipy import stats


@tool
def run_statistical_tests_tool(
    dataset_path: str,
    target_column: str,
    max_categories: int = 5,
) -> Dict[str, Any]:
    """
    Run basic statistical tests between features and target. Returns summarized results.
    """
    df = pd.read_csv(dataset_path)
    if target_column not in df.columns:
        return {"error": "Target column not found"}

    target = df[target_column]
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    results: Dict[str, Any] = {
        "t_tests": [],
        "mann_whitney": [],
        "anova": [],
        "chi_square": [],
        "correlations": [],
    }

    # Correlation significance (numeric vs target if target is numeric)
    if target_column in numeric_cols:
        for col in numeric_cols:
            if col == target_column:
                continue
            try:
                corr, p_val = stats.pearsonr(df[col], target)
                results["correlations"].append(
                    {"feature": col, "corr": float(corr), "p_value": float(p_val)}
                )
            except Exception:
                continue

    # Binary target tests
    if target.nunique() == 2:
        groups = [df[target == label] for label in sorted(target.unique())]

        for col in numeric_cols:
            if col == target_column:
                continue
            try:
                t_stat, p_val = stats.ttest_ind(
                    groups[0][col], groups[1][col], nan_policy="omit"
                )
                results["t_tests"].append(
                    {"feature": col, "t_stat": float(t_stat), "p_value": float(p_val)}
                )
                u_stat, p_u = stats.mannwhitneyu(
                    groups[0][col], groups[1][col], alternative="two-sided"
                )
                results["mann_whitney"].append(
                    {"feature": col, "u_stat": float(u_stat), "p_value": float(p_u)}
                )
            except Exception:
                continue

    # ANOVA for numeric features vs categorical target with >2 classes
    if target.nunique() > 2:
        for col in numeric_cols:
            if col == target_column:
                continue
            try:
                grouped = [
                    df[df[target_column] == label][col] for label in target.unique()
                ]
                f_stat, p_val = stats.f_oneway(*grouped)
                results["anova"].append(
                    {"feature": col, "f_stat": float(f_stat), "p_value": float(p_val)}
                )
            except Exception:
                continue

    # Chi-square for categorical features vs target
    for col in categorical_cols:
        if col == target_column:
            continue
        if df[col].nunique() > max_categories:
            continue
        try:
            contingency = pd.crosstab(df[col], target)
            chi2, p_val, _, _ = stats.chi2_contingency(contingency)
            results["chi_square"].append(
                {"feature": col, "chi2": float(chi2), "p_value": float(p_val)}
            )
        except Exception:
            continue

    return results
