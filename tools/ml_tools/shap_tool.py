from __future__ import annotations

from typing import Any, Dict, List

import os
import joblib
import pandas as pd
import shap
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from langchain.tools import tool


@tool
def compute_shap_summary_tool(
    model_path: str,
    dataset_path: str,
    target_column: str,
    output_dir: str = "outputs/visualizations",
    sample_size: int = 200,
) -> Dict[str, Any]:
    """
    Compute SHAP summary plot and top features from a saved model pipeline.
    Returns only summarized results and image paths (no raw data).
    """
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(dataset_path)
    if target_column not in df.columns:
        return {"error": "Target column not found."}

    X = df.drop(columns=[target_column])
    if len(X) > sample_size:
        X = X.sample(n=sample_size, random_state=42)

    model = joblib.load(model_path)

    try:
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        shap.summary_plot(shap_values, X, show=False, plot_type="bar")
        plot_path = os.path.join(output_dir, "shap_summary.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()

        mean_abs = shap_values.abs.mean(0).values
        feature_names = list(X.columns)
        ranking = sorted(
            zip(feature_names, mean_abs.tolist()), key=lambda x: x[1], reverse=True
        )

        top_features = [
            {"feature": name, "importance": float(score)}
            for name, score in ranking[:10]
        ]

        return {
            "plot_path": plot_path,
            "top_features": top_features,
        }
    except Exception as exc:
        return {"error": f"SHAP failed: {exc}"}
