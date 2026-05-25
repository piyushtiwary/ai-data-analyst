from __future__ import annotations

from typing import Any, Dict, List

import os
import joblib
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from langchain.tools import tool

from sklearn.metrics import confusion_matrix, RocCurveDisplay


@tool
def build_visualizations_tool(
    dataset_path: str,
    target_column: str,
    model_path: str,
    visualization_plan: list[str],
    output_dir: str = "outputs/visualizations",
) -> Dict[str, Any]:
    """
    Build visualization assets based on metadata and model outputs only.
    """
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(dataset_path)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    model = joblib.load(model_path)

    output_paths: Dict[str, str] = {}

    if "class_distribution" in visualization_plan:
        plt.figure(figsize=(6, 4))
        sns.countplot(x=y)
        plt.title("Class Distribution")
        class_path = os.path.join(output_dir, "class_distribution.png")
        plt.tight_layout()
        plt.savefig(class_path, dpi=150)
        plt.close()
        output_paths["class_distribution"] = class_path

    if "confusion_matrix" in visualization_plan:
        preds = model.predict(X)
        cm = confusion_matrix(y, preds)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.tight_layout()
        plt.savefig(cm_path, dpi=150)
        plt.close()
        output_paths["confusion_matrix"] = cm_path

    if "roc_curve" in visualization_plan:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[:, 1]
            RocCurveDisplay.from_predictions(y, proba)
            plt.title("ROC Curve")
            roc_path = os.path.join(output_dir, "roc_curve.png")
            plt.tight_layout()
            plt.savefig(roc_path, dpi=150)
            plt.close()
            output_paths["roc_curve"] = roc_path

    if "feature_importance" in visualization_plan:
        feature_importance = {}
        if hasattr(model, "named_steps") and "model" in model.named_steps:
            estimator = model.named_steps["model"]
            if hasattr(estimator, "feature_importances_"):
                try:
                    feature_names = model.named_steps[
                        "preprocessor"
                    ].get_feature_names_out()
                    feature_importance = dict(
                        zip(
                            feature_names.tolist(),
                            estimator.feature_importances_.tolist(),
                        )
                    )
                except Exception:
                    feature_importance = {}
            elif hasattr(estimator, "coef_"):
                try:
                    feature_names = model.named_steps[
                        "preprocessor"
                    ].get_feature_names_out()
                    coefficients = estimator.coef_[0]
                    feature_importance = dict(
                        zip(
                            feature_names.tolist(),
                            [abs(v) for v in coefficients.tolist()],
                        )
                    )
                except Exception:
                    feature_importance = {}

        if feature_importance:
            top_items = list(feature_importance.items())[:15]
            labels = [item[0] for item in top_items]
            values = [item[1] for item in top_items]
            plt.figure(figsize=(8, 5))
            sns.barplot(x=values, y=labels)
            plt.title("Top Feature Importance")
            fi_path = os.path.join(output_dir, "feature_importance.png")
            plt.tight_layout()
            plt.savefig(fi_path, dpi=150)
            plt.close()
            output_paths["feature_importance"] = fi_path

    if "correlation_heatmap" in visualization_plan:
        numeric_df = df.select_dtypes(include=["number"])
        if not numeric_df.empty:
            plt.figure(figsize=(8, 6))
            sns.heatmap(numeric_df.corr(numeric_only=True), cmap="coolwarm", center=0)
            plt.title("Correlation Heatmap")
            corr_path = os.path.join(output_dir, "correlation_heatmap.png")
            plt.tight_layout()
            plt.savefig(corr_path, dpi=150)
            plt.close()
            output_paths["correlation_heatmap"] = corr_path

    if "pairplot" in visualization_plan:
        focus_pairs = [
            ("sleep_hours", target_column),
            ("stress_level", target_column),
            ("anxiety_level", target_column),
        ]

        for feature, target in focus_pairs:
            if feature in df.columns and target in df.columns:
                plt.figure(figsize=(6, 4))
                sns.stripplot(x=df[target], y=df[feature], jitter=0.25, alpha=0.6)
                plt.title(f"{feature} vs {target}")
                plot_path = os.path.join(output_dir, f"focus_{feature}_vs_{target}.png")
                plt.tight_layout()
                plt.savefig(plot_path, dpi=150)
                plt.close()
                output_paths[f"focus_{feature}_vs_{target}"] = plot_path

    if "numerical_distributions" in visualization_plan:
        numeric_df = df.select_dtypes(include=["number"]).copy()
        if not numeric_df.empty:
            plot_path = os.path.join(output_dir, "numerical_distributions.png")
            numeric_df.hist(figsize=(10, 8), bins=20)
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150)
            plt.close("all")
            output_paths["numerical_distributions"] = plot_path

    if "categorical_distributions" in visualization_plan:
        if categorical_cols := df.select_dtypes(include=["object"]).columns.tolist():
            for col in categorical_cols[:5]:
                plt.figure(figsize=(6, 4))
                sns.countplot(x=df[col])
                plt.title(f"{col} Distribution")
                cat_path = os.path.join(output_dir, f"categorical_{col}.png")
                plt.tight_layout()
                plt.savefig(cat_path, dpi=150)
                plt.close()
                output_paths[f"categorical_{col}"] = cat_path

    return {
        "output_dir": output_dir,
        "assets": output_paths,
    }
