from typing import Any, Dict, List, cast
from pathlib import Path

from agents.ml_agent import ml_planning_chain, MLPlanningOutput
from agents.ml_reasoning_agent import ml_reasoning_chain, MLReasoningOutput
from agents.validation_agent import validation_chain, ValidationOutput
from agents.report_agent import report_chain, ReportOutput
from agents.eda_reasoning_agent import eda_reasoning_chain, EDANarrativeOutput
from agents.business_insights_agent import (
    business_insights_chain,
    BusinessInsightsOutput,
)
from agents.visualization_narrative_agent import (
    visualization_narrative_chain,
    VisualizationNarrativeOutput,
)

from tools.ml_tools import (
    XGBoost,
    LogisticRegression,
    RandomForest,
    SVC,
    KNN,
    class_balance,
)
from tools.ml_tools import thresholds, evidence, experiment_store, shap_tool
from tools.ml_tools import data_quality, stat_tests
from tools.visualization_tools import visualization_builder
from tools.report_tools import html_report

from states.root_state import RootState


def _extract_ordinal_columns(
    encoding_strategy: list[Dict[str, str]] | None,
) -> list[str]:
    if not encoding_strategy:
        return []

    return [
        rule.get("column_name")
        for rule in encoding_strategy
        if rule.get("encoding_type") == "ordinal"
    ]


def _compact_experiments(experiments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compact = []
    for exp in experiments:
        compact.append(
            {
                "model_name": exp.get("model_name"),
                "metrics": exp.get("metrics", {}),
                "cv_mean_roc_auc": exp.get("cv_mean_roc_auc"),
                "cv_std_roc_auc": exp.get("cv_std_roc_auc"),
                "fold_scores": exp.get("fold_scores", []),
            }
        )

    return compact


def _top_feature_importance(feature_importance: Dict[str, float], top_k: int = 10):
    if not feature_importance:
        return {}

    items = list(feature_importance.items())[:top_k]
    return {key: value for key, value in items}


def _condense_statistical_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    condensed: Dict[str, Any] = {}
    for column, stats in summary.items():
        if not isinstance(stats, dict):
            continue

        condensed[column] = {
            key: stats.get(key)
            for key in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
            if key in stats
        }

    return condensed


def _format_visualization_assets(assets: Dict[str, str]) -> List[Dict[str, str]]:
    formatted = []
    for name, path in assets.items():
        uri = Path(path).resolve().as_uri()
        formatted.append({"name": name, "path": uri})

    return formatted


async def analyze_class_balance(state: RootState):
    data_state = state["data"]
    ml_state = state["ml"]

    tool_result = await class_balance.analyze_class_balance_tool.ainvoke(
        {
            "dataset_path": data_state["clean_dataset_path"],
            "target_column": data_state["probable_target_column"],
        }
    )

    return {
        "ml": {
            **ml_state,
            "class_balance": tool_result,
        }
    }


async def data_quality_node(state: RootState):
    data_state = state["data"]
    report_state = state.get("report", {})

    tool_result = await data_quality.assess_data_quality_tool.ainvoke(
        {
            "dataset_path": data_state["clean_dataset_path"],
            "target_column": data_state["probable_target_column"],
        }
    )

    return {
        "report": {
            **report_state,
            "data_quality_assessment": tool_result,
        }
    }


async def statistical_tests_node(state: RootState):
    data_state = state["data"]
    report_state = state.get("report", {})

    tool_result = await stat_tests.run_statistical_tests_tool.ainvoke(
        {
            "dataset_path": data_state["clean_dataset_path"],
            "target_column": data_state["probable_target_column"],
        }
    )

    return {
        "report": {
            **report_state,
            "statistical_tests": tool_result,
        }
    }


async def eda_reasoning_node(state: RootState):
    data_state = state["data"]
    ml_state = state["ml"]
    report_state = state.get("report", {})

    feature_corrs = ml_state.get("evaluation_evidence", {}).get(
        "feature_target_correlations", []
    )

    eda_output = cast(
        EDANarrativeOutput,
        await eda_reasoning_chain.ainvoke(
            {
                "dataset_summary": data_state.get("dataset_summary", ""),
                "statistical_summary": _condense_statistical_summary(
                    data_state.get("statistical_summary", {})
                ),
                "class_balance": ml_state.get("class_balance", {}),
                "feature_target_correlations": feature_corrs,
                "statistical_tests": report_state.get("statistical_tests", {}),
                "data_quality": report_state.get("data_quality_assessment", {}),
            }
        ),
    )

    return {
        "report": {
            **report_state,
            "eda_narrative": eda_output.eda_narrative,
            "hypothesis_list": eda_output.hypotheses,
        }
    }


async def business_insights_node(state: RootState):
    ml_state = state["ml"]
    report_state = state.get("report", {})

    insights_output = cast(
        BusinessInsightsOutput,
        await business_insights_chain.ainvoke(
            {
                "eda_narrative": report_state.get("eda_narrative", ""),
                "statistical_tests": report_state.get("statistical_tests", {}),
                "class_balance": ml_state.get("class_balance", {}),
                "risk_metrics": ml_state.get("risk_metrics", {}),
            }
        ),
    )

    return {
        "report": {
            **report_state,
            "business_insights": insights_output.insights,
        }
    }


async def ml_planning_node(state: RootState):
    data_state = state["data"]
    ml_state = state["ml"]

    planning_output = cast(
        MLPlanningOutput,
        await ml_planning_chain.ainvoke(
            {
                "dataset_summary": data_state.get("dataset_summary", ""),
                "problem_type": data_state.get("problem_type", ""),
                "evaluation_metrics": data_state.get("evaluation_metrics", []),
                "class_balance": ml_state.get("class_balance", {}),
            }
        ),
    )

    plan_dict = planning_output.model_dump()
    selected_models = plan_dict.get("selected_models") or data_state.get(
        "selected_models", []
    )

    primary_metric = plan_dict.get("primary_metric") or "f1_score"
    retry_count = ml_state.get("retry_count", 0)
    max_retries = ml_state.get("max_retries", 2)
    cv_folds = ml_state.get("cv_folds", 5)
    cv_repeats = ml_state.get("cv_repeats", 2)
    cv_std_threshold = ml_state.get("cv_std_threshold", 0.05)
    cv_mean_threshold = ml_state.get("cv_mean_threshold", 0.75)
    correlation_threshold = ml_state.get("correlation_threshold", 0.9)
    duplicate_overlap_threshold = ml_state.get("duplicate_overlap_threshold", 0.0)

    return {
        "ml": {
            **ml_state,
            **plan_dict,
            "selected_models": selected_models,
            "primary_metric": primary_metric,
            "retry_count": retry_count,
            "max_retries": max_retries,
            "cv_folds": cv_folds,
            "cv_repeats": cv_repeats,
            "cv_std_threshold": cv_std_threshold,
            "cv_mean_threshold": cv_mean_threshold,
            "correlation_threshold": correlation_threshold,
            "duplicate_overlap_threshold": duplicate_overlap_threshold,
            "evaluation_metrics": data_state.get("evaluation_metrics", []),
        }
    }


async def train_selected_models(state: RootState):
    data_state = state["data"]
    ml_state = state["ml"]

    selected_models = ml_state.get("selected_models", [])
    if not selected_models:
        selected_models = [
            "LogisticRegression",
            "RandomForestClassifier",
            "XGBoostClassifier",
        ]

    ordinal_columns = _extract_ordinal_columns(data_state.get("encoding_strategy"))

    tool_payload = {
        "dataset_path": data_state["clean_dataset_path"],
        "target_column": data_state["probable_target_column"],
        "categorical_columns": data_state.get("categorical_columns", []),
        "numerical_columns": data_state.get("numerical_columns", []),
        "ordinal_columns": ordinal_columns,
        "scaling_required": bool(data_state.get("scaling_required", False)),
        "imbalance_strategy": ml_state.get("imbalance_strategy", "none"),
        "cv_folds": ml_state.get("cv_folds", 5),
        "cv_repeats": ml_state.get("cv_repeats", 2),
    }

    tool_map = {
        "LogisticRegression": LogisticRegression.train_logistic_regression_tool,
        "RandomForestClassifier": RandomForest.train_random_forest_tool,
        "XGBoostClassifier": XGBoost.train_xgboost_classifier_tool,
        "SVC": SVC.train_svc_tool,
        "KNeighborsClassifier": KNN.train_knn_tool,
    }

    experiments: List[Dict[str, Any]] = [*ml_state.get("experiments", [])]

    for model_name in selected_models:
        tool = tool_map.get(model_name)
        if tool is None:
            continue

        tool_result = await tool.ainvoke(tool_payload)
        await experiment_store.append_experiment_tool.ainvoke(
            {
                "experiment": tool_result,
            }
        )
        experiments.append(tool_result)

    return {
        "ml": {
            **ml_state,
            "experiments": experiments,
        }
    }


async def load_experiment_history(state: RootState):
    ml_state = state["ml"]

    result = await experiment_store.load_experiment_history_tool.ainvoke({})

    return {
        "ml": {
            **ml_state,
            "experiment_history": result.get("experiments", []),
        }
    }


async def select_best_model(state: RootState):
    ml_state = state["ml"]
    experiments = ml_state.get("experiments", [])

    if not experiments:
        return {"ml": ml_state}

    primary_metric = ml_state.get("primary_metric", "f1_score")

    def metric_value(exp: Dict[str, Any], key: str) -> float:
        metrics = exp.get("metrics", {})
        return float(metrics.get(key, 0.0))

    best_experiment = max(
        experiments,
        key=lambda exp: (
            metric_value(exp, primary_metric),
            metric_value(exp, "roc_auc"),
        ),
    )

    best_metric = metric_value(best_experiment, primary_metric)
    best_roc_auc = metric_value(best_experiment, "roc_auc")
    cv_mean_roc_auc = float(best_experiment.get("cv_mean_roc_auc", 0.0) or 0.0)
    cv_std_roc_auc = float(best_experiment.get("cv_std_roc_auc", 0.0) or 0.0)

    reason = (
        f"Selected {best_experiment.get('model_name')} with {primary_metric}="
        f"{best_metric:.4f} and roc_auc={best_roc_auc:.4f}. "
        f"CV mean roc_auc={cv_mean_roc_auc:.4f}, std={cv_std_roc_auc:.4f}."
    )
    return {
        "ml": {
            **ml_state,
            "best_model_name": best_experiment.get("model_name"),
            "best_model_path": best_experiment.get("model_path"),
            "best_model_metrics": best_experiment.get("metrics"),
            "best_model_hyperparameters": best_experiment.get("hyperparameters"),
            "best_model_feature_importance": best_experiment.get(
                "feature_importance", {}
            ),
            "best_model_cv_mean_roc_auc": cv_mean_roc_auc,
            "best_model_cv_std_roc_auc": cv_std_roc_auc,
            "best_model_reason": reason,
        }
    }


async def validation_node(state: RootState):
    ml_state = state["ml"]

    experiments = _compact_experiments(ml_state.get("experiments", []))

    validation_output = cast(
        ValidationOutput,
        await validation_chain.ainvoke(
            {
                "experiments": experiments,
                "best_model": ml_state.get("best_model_name"),
                "best_model_metrics": ml_state.get("best_model_metrics", {}),
                "cv_stability": {
                    "cv_mean_roc_auc": ml_state.get("best_model_cv_mean_roc_auc"),
                    "cv_std_roc_auc": ml_state.get("best_model_cv_std_roc_auc"),
                },
                "evaluation_evidence": ml_state.get("evaluation_evidence", {}),
                "class_balance": ml_state.get("class_balance", {}),
            }
        ),
    )

    validation_dict = validation_output.model_dump()

    return {
        "ml": {
            **ml_state,
            **validation_dict,
        }
    }


async def prepare_retry(state: RootState):
    ml_state = state["ml"]
    retry_count = int(ml_state.get("retry_count", 0)) + 1
    selected_models = list(ml_state.get("selected_models", []))
    best_model = ml_state.get("best_model_name")

    if best_model in selected_models:
        selected_models.remove(best_model)

    return {
        "ml": {
            **ml_state,
            "retry_count": retry_count,
            "selected_models": selected_models,
            "experiments": [],
            "evaluation_evidence": {},
            "performance_metrics": {},
            "stability_metrics": {},
            "risk_metrics": {},
            "business_metrics": {},
            "best_model_name": None,
            "best_model_path": None,
            "best_model_metrics": None,
            "best_model_hyperparameters": None,
            "best_model_feature_importance": None,
            "best_model_reason": None,
            "cv_folds": int(ml_state.get("cv_folds", 5)) + 1,
            "cv_repeats": int(ml_state.get("cv_repeats", 2)) + 1,
            "cv_std_threshold": max(
                float(ml_state.get("cv_std_threshold", 0.05)) - 0.01, 0.01
            ),
            "cv_mean_threshold": min(
                float(ml_state.get("cv_mean_threshold", 0.75)) + 0.02, 0.95
            ),
            "correlation_threshold": max(
                float(ml_state.get("correlation_threshold", 0.9)) - 0.05, 0.7
            ),
            "duplicate_overlap_threshold": 0.0,
        }
    }


def decide_after_validation(state: RootState) -> str:
    ml_state = state["ml"]
    leakage_risk = bool(ml_state.get("leakage_risk", False))
    retry_count = int(ml_state.get("retry_count", 0))
    max_retries = int(ml_state.get("max_retries", 2))

    cv_std = float(ml_state.get("best_model_cv_std_roc_auc", 0.0) or 0.0)
    cv_mean = float(ml_state.get("best_model_cv_mean_roc_auc", 0.0) or 0.0)
    cv_std_threshold = float(ml_state.get("cv_std_threshold", 0.05))
    cv_mean_threshold = float(ml_state.get("cv_mean_threshold", 0.75))
    evidence = ml_state.get("evaluation_evidence", {})
    duplicate_overlap = float(evidence.get("duplicate_overlap", 0.0) or 0.0)
    max_corr = float(evidence.get("feature_target_correlation_max", 0.0) or 0.0)
    duplicate_threshold = float(ml_state.get("duplicate_overlap_threshold", 0.0))
    corr_threshold = float(ml_state.get("correlation_threshold", 0.9))

    selected_models = ml_state.get("selected_models", [])
    has_alternatives = len(selected_models) > 1

    if (
        (
            leakage_risk
            or cv_std > cv_std_threshold
            or cv_mean < cv_mean_threshold
            or duplicate_overlap > duplicate_threshold
            or max_corr > corr_threshold
        )
        and retry_count < max_retries
        and has_alternatives
    ):
        return "retry"

    return "proceed"


async def ml_reasoning_node(state: RootState):
    ml_state = state["ml"]

    experiments = _compact_experiments(ml_state.get("experiments", []))
    feature_importance = _top_feature_importance(
        ml_state.get("best_model_feature_importance", {})
    )

    reasoning_output = cast(
        MLReasoningOutput,
        await ml_reasoning_chain.ainvoke(
            {
                "experiments": experiments,
                "class_balance": ml_state.get("class_balance", {}),
                "best_model": ml_state.get("best_model_name"),
                "best_model_metrics": ml_state.get("best_model_metrics", {}),
                "feature_importance": feature_importance,
                "leakage_risk": ml_state.get("leakage_risk", False),
            }
        ),
    )

    reasoning_dict = reasoning_output.model_dump()

    return {
        "ml": {
            **ml_state,
            **reasoning_dict,
        }
    }


async def build_ml_report(state: RootState):
    data_state = state["data"]
    ml_state = state["ml"]
    report_state = state.get("report", {})

    class_balance = ml_state.get("class_balance", {})
    minority_rate = class_balance.get("minority_rate")
    imbalance_ratio = class_balance.get("imbalance_ratio")
    imbalance_strategy = ml_state.get("imbalance_strategy", "none")
    primary_metric = ml_state.get("primary_metric", "f1_score")

    insights = list(report_state.get("insights", []))
    recommendations = list(report_state.get("recommendations", []))

    if minority_rate is not None:
        insights.append(
            "Class balance: minority rate="
            f"{minority_rate:.4f}, imbalance ratio={imbalance_ratio:.2f}."
        )

    insights.append(f"Imbalance strategy applied: {imbalance_strategy}.")

    best_model_name = ml_state.get("best_model_name")
    best_model_reason = ml_state.get("best_model_reason")
    leakage_risk = ml_state.get("leakage_risk")
    leakage_reasons = ml_state.get("leakage_reasons", [])
    leakage_recommendations = ml_state.get("leakage_recommendations", [])
    next_action = ml_state.get("next_action")
    tuning_required = ml_state.get("tuning_required")
    recommended_tuning = ml_state.get("recommended_tuning")
    threshold_optimization = ml_state.get("threshold_optimization", {})
    cv_mean = ml_state.get("best_model_cv_mean_roc_auc")
    cv_std = ml_state.get("best_model_cv_std_roc_auc")

    if best_model_name:
        insights.append(f"Best model: {best_model_name}.")

    if best_model_reason:
        insights.append(best_model_reason)

    if primary_metric:
        insights.append(f"Primary metric: {primary_metric}.")

    if cv_mean is not None and cv_std is not None:
        insights.append(
            f"Cross-validation roc_auc mean={cv_mean:.4f}, std={cv_std:.4f}."
        )

    if leakage_risk:
        insights.append("Leakage risk detected based on validation heuristics.")
        insights.extend(leakage_reasons)
        recommendations.extend(leakage_recommendations)

    if next_action:
        recommendations.append(f"Next action: {next_action}.")

    if tuning_required:
        recommendations.append("Hyperparameter tuning recommended.")

    if recommended_tuning:
        model_name = recommended_tuning.get("model")
        params = recommended_tuning.get("params")
        if model_name and params:
            recommendations.append(f"Tuning plan: {recommended_tuning}.")

    if threshold_optimization:
        recommendations.append("Apply optimized decision thresholds for deployment.")
        best_f1_threshold = threshold_optimization.get("best_f1_threshold")
        best_f1 = threshold_optimization.get("best_f1")
        if best_f1_threshold is not None and best_f1 is not None:
            insights.append(
                f"Best F1 threshold={best_f1_threshold:.2f} with f1={best_f1:.4f}."
            )

    validation_checks = ml_state.get("validation_checks", {})
    evidence_summary = ml_state.get("evidence_summary", [])

    if validation_checks:
        insights.append(f"Validation checks: {validation_checks}.")

    if evidence_summary:
        insights.extend(evidence_summary)
    if data_state.get("evaluation_metrics"):
        recommendations.append(
            "Track evaluation metrics over multiple splits for stability checks."
        )

    model_comparison = []
    for exp in ml_state.get("experiments", []):
        metrics = exp.get("metrics", {})
        cv_mean_roc_auc = exp.get("cv_mean_roc_auc", 0.0)
        cv_std_roc_auc = exp.get("cv_std_roc_auc", 0.0)
        model_comparison.append(
            f"{exp.get('model_name')}: {primary_metric}="
            f"{metrics.get(primary_metric, 0.0):.4f}, roc_auc="
            f"{metrics.get('roc_auc', 0.0):.4f}, cv_mean_roc_auc="
            f"{cv_mean_roc_auc:.4f}, cv_std_roc_auc={cv_std_roc_auc:.4f}"
        )

    ml_summary = "\n".join(insights)

    return {
        "report": {
            **report_state,
            "insights": insights,
            "recommendations": recommendations,
            "ml_summary": ml_summary,
            "model_comparison": model_comparison,
        }
    }


async def evidence_validation_node(state: RootState):
    data_state = state["data"]
    ml_state = state["ml"]

    if not ml_state.get("best_model_path"):
        return {"ml": ml_state}

    evidence_result = await evidence.evaluate_model_evidence_tool.ainvoke(
        {
            "model_path": ml_state["best_model_path"],
            "dataset_path": data_state["clean_dataset_path"],
            "target_column": data_state["probable_target_column"],
            "correlation_threshold": ml_state.get("correlation_threshold", 0.9),
            "duplicate_overlap_threshold": ml_state.get(
                "duplicate_overlap_threshold", 0.0
            ),
        }
    )

    evidence_result["cv_mean"] = ml_state.get("best_model_cv_mean_roc_auc")
    evidence_result["cv_std"] = ml_state.get("best_model_cv_std_roc_auc")

    performance_metrics = {
        "roc_auc": float(ml_state.get("best_model_metrics", {}).get("roc_auc", 0.0)),
        "f1_score": float(ml_state.get("best_model_metrics", {}).get("f1_score", 0.0)),
        "accuracy": float(ml_state.get("best_model_metrics", {}).get("accuracy", 0.0)),
    }

    stability_metrics = {
        "cv_mean_roc_auc": float(evidence_result.get("cv_mean") or 0.0),
        "cv_std_roc_auc": float(evidence_result.get("cv_std") or 0.0),
        "train_score": float(evidence_result.get("train_score", 0.0)),
        "test_score": float(evidence_result.get("test_score", 0.0)),
    }

    risk_metrics = {
        "duplicate_overlap": float(evidence_result.get("duplicate_overlap", 0.0)),
        "feature_target_correlation_max": float(
            evidence_result.get("feature_target_correlation_max", 0.0)
        ),
        "duplicate_overlap_ok": bool(
            evidence_result.get("duplicate_overlap_ok", False)
        ),
        "correlation_ok": bool(evidence_result.get("correlation_ok", False)),
        "preprocessing_contamination": evidence_result.get(
            "preprocessing_contamination", "unknown"
        ),
    }

    business_metrics = {
        "class_imbalance_ratio": float(
            ml_state.get("class_balance", {}).get("imbalance_ratio", 0.0)
        ),
        "minority_rate": float(
            ml_state.get("class_balance", {}).get("minority_rate", 0.0)
        ),
        "best_f1_threshold": ml_state.get("threshold_optimization", {}).get(
            "best_f1_threshold"
        ),
    }

    return {
        "ml": {
            **ml_state,
            "evaluation_evidence": evidence_result,
            "performance_metrics": performance_metrics,
            "stability_metrics": stability_metrics,
            "risk_metrics": risk_metrics,
            "business_metrics": business_metrics,
        }
    }


async def shap_analysis_node(state: RootState):
    data_state = state["data"]
    ml_state = state["ml"]
    report_state = state.get("report", {})

    if not ml_state.get("best_model_path"):
        return {"report": report_state}

    shap_result = await shap_tool.compute_shap_summary_tool.ainvoke(
        {
            "model_path": ml_state["best_model_path"],
            "dataset_path": data_state["clean_dataset_path"],
            "target_column": data_state["probable_target_column"],
        }
    )

    visualization_assets = dict(report_state.get("visualization_assets", {}))
    if shap_result.get("plot_path"):
        visualization_assets["shap_summary"] = shap_result["plot_path"]

    return {
        "report": {
            **report_state,
            "visualization_assets": visualization_assets,
            "shap_summary": shap_result,
        }
    }


async def visualization_narrative_node(state: RootState):
    ml_state = state["ml"]
    report_state = state.get("report", {})

    assets = _format_visualization_assets(report_state.get("visualization_assets", {}))

    narrative_output = cast(
        VisualizationNarrativeOutput,
        await visualization_narrative_chain.ainvoke(
            {
                "visualization_assets": assets,
                "class_balance": ml_state.get("class_balance", {}),
                "performance_metrics": ml_state.get("performance_metrics", {}),
                "stability_metrics": ml_state.get("stability_metrics", {}),
                "validation_flags": {
                    "leakage_risk": ml_state.get("leakage_risk", False),
                    "validation_checks": ml_state.get("validation_checks", {}),
                },
                "shap_summary": report_state.get("shap_summary", {}),
            }
        ),
    )

    return {
        "report": {
            **report_state,
            "visualization_narrative": narrative_output.narratives,
        }
    }


async def threshold_optimization_node(state: RootState):
    data_state = state["data"]
    ml_state = state["ml"]

    if not ml_state.get("best_model_path"):
        return {"ml": ml_state}

    tool_result = await thresholds.optimize_thresholds_tool.ainvoke(
        {
            "model_path": ml_state["best_model_path"],
            "dataset_path": data_state["clean_dataset_path"],
            "target_column": data_state["probable_target_column"],
        }
    )

    return {
        "ml": {
            **ml_state,
            "threshold_optimization": tool_result,
        }
    }


async def build_visualizations_node(state: RootState):
    data_state = state["data"]
    ml_state = state["ml"]
    report_state = state.get("report", {})

    if not ml_state.get("best_model_path"):
        return {"report": report_state}

    visualization_plan = data_state.get("visualization_plan", [])
    tool_result = await visualization_builder.build_visualizations_tool.ainvoke(
        {
            "dataset_path": data_state["clean_dataset_path"],
            "target_column": data_state["probable_target_column"],
            "model_path": ml_state["best_model_path"],
            "visualization_plan": visualization_plan,
        }
    )

    merged_assets = dict(report_state.get("visualization_assets", {}))
    merged_assets.update(tool_result.get("assets", {}))

    return {
        "report": {
            **report_state,
            "visualization_assets": merged_assets,
        }
    }


async def generate_pdf_report_node(state: RootState):
    data_state = state["data"]
    ml_state = state["ml"]
    report_state = state.get("report", {})

    condensed_stats = _condense_statistical_summary(
        data_state.get("statistical_summary", {})
    )

    visualization_assets = _format_visualization_assets(
        report_state.get("visualization_assets", {})
    )

    report_output = cast(
        ReportOutput,
        await report_chain.ainvoke(
            {
                "dataset_summary": data_state.get("dataset_summary", ""),
                "business_domain": data_state.get("business_domain", ""),
                "shape": data_state.get("shape", ""),
                "columns": data_state.get("columns", []),
                "dtypes": data_state.get("dtypes", {}),
                "target_column": data_state.get("probable_target_column", ""),
                "problem_type": data_state.get("problem_type", ""),
                "raw_missing_summary": data_state.get("raw_missing_summary", {}),
                "clean_missing_summary": data_state.get("clean_missing_summary", {}),
                "statistical_summary": condensed_stats,
                "encoding_strategy": data_state.get("encoding_strategy", []),
                "scaling_required": data_state.get("scaling_required", False),
                "class_balance": ml_state.get("class_balance", {}),
                "recommended_tasks": data_state.get("recommended_tasks", []),
                "ml_summary": report_state.get("ml_summary", ""),
                "performance_metrics": ml_state.get("performance_metrics", {}),
                "stability_metrics": ml_state.get("stability_metrics", {}),
                "risk_metrics": ml_state.get("risk_metrics", {}),
                "business_metrics": ml_state.get("business_metrics", {}),
                "model_comparison": report_state.get("model_comparison", []),
                "threshold_optimization": ml_state.get("threshold_optimization", {}),
                "validation_flags": {
                    "leakage_risk": ml_state.get("leakage_risk", False),
                    "leakage_reasons": ml_state.get("leakage_reasons", []),
                    "leakage_recommendations": ml_state.get(
                        "leakage_recommendations", []
                    ),
                    "validation_checks": ml_state.get("validation_checks", {}),
                    "evidence_summary": ml_state.get("evidence_summary", []),
                },
                "evaluation_evidence": ml_state.get("evaluation_evidence", {}),
                "eda_narrative": report_state.get("eda_narrative", ""),
                "hypotheses": report_state.get("hypothesis_list", []),
                "statistical_tests": report_state.get("statistical_tests", {}),
                "data_quality_assessment": report_state.get(
                    "data_quality_assessment", {}
                ),
                "business_insights": report_state.get("business_insights", []),
                "visualization_narrative": report_state.get(
                    "visualization_narrative", []
                ),
                "visualization_assets": visualization_assets,
                "shap_summary": report_state.get("shap_summary", {}),
            }
        ),
    )

    report_dict = report_output.model_dump()

    html_result = await html_report.generate_pdf_from_html_tool.ainvoke(
        {
            "output_pdf_path": "outputs/report.pdf",
            "output_html_path": "outputs/report.html",
            "title": report_dict.get("title", "ML Analysis Report"),
            "html_content": report_dict.get("html_content", ""),
        }
    )

    return {
        "report": {
            **report_state,
            **html_result,
        }
    }
