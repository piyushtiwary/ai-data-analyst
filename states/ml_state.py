from typing_extensions import TypedDict, NotRequired
from typing import Any, Dict


class ExperimentState(TypedDict):

    experiment_id: str

    model_name: str

    model_path: str

    metrics: Dict[str, float]

    hyperparameters: Dict[str, Any]

    feature_importance: Dict[str, float]

    cv_mean_roc_auc: NotRequired[float]

    cv_std_roc_auc: NotRequired[float]

    fold_scores: NotRequired[list[float]]

    performance_metrics: NotRequired[Dict[str, float]]

    stability_metrics: NotRequired[Dict[str, float]]

    risk_metrics: NotRequired[Dict[str, Any]]

    business_metrics: NotRequired[Dict[str, Any]]


class MLState(TypedDict):

    selected_models: list[str]

    experiments: list[ExperimentState]

    class_balance: NotRequired[Dict[str, Any]]

    imbalance_strategy: NotRequired[str]

    imbalance_reason: NotRequired[str]

    primary_metric: NotRequired[str]

    evaluation_metrics: NotRequired[list[str]]

    best_model_name: NotRequired[str]

    best_model_path: NotRequired[str]

    best_model_metrics: NotRequired[dict]

    best_model_hyperparameters: NotRequired[dict]

    best_model_feature_importance: NotRequired[Dict[str, float]]

    best_model_cv_mean_roc_auc: NotRequired[float]

    best_model_cv_std_roc_auc: NotRequired[float]

    best_model_reason: NotRequired[str]

    leakage_risk: NotRequired[bool]

    leakage_reasons: NotRequired[list[str]]

    leakage_recommendations: NotRequired[list[str]]

    evidence_summary: NotRequired[list[str]]

    validation_checks: NotRequired[dict]

    current_model: NotRequired[str]

    tuning_required: NotRequired[bool]

    cv_folds: NotRequired[int]

    cv_repeats: NotRequired[int]

    cv_std_threshold: NotRequired[float]

    cv_mean_threshold: NotRequired[float]

    retry_count: NotRequired[int]

    max_retries: NotRequired[int]

    threshold_optimization: NotRequired[Dict[str, Any]]

    evaluation_evidence: NotRequired[Dict[str, Any]]

    performance_metrics: NotRequired[Dict[str, float]]

    stability_metrics: NotRequired[Dict[str, float]]

    risk_metrics: NotRequired[Dict[str, Any]]

    business_metrics: NotRequired[Dict[str, Any]]

    correlation_threshold: NotRequired[float]

    duplicate_overlap_threshold: NotRequired[float]

    experiment_history: NotRequired[list[Dict[str, Any]]]

    next_action: NotRequired[str]

    recommended_tuning: NotRequired[dict]
