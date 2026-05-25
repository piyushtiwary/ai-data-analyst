from __future__ import annotations

from typing import List

from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate

from .models import openai_model


class MLPlanningOutput(BaseModel):
    selected_models: List[str]
    imbalance_strategy: str
    imbalance_reason: str
    primary_metric: str
    tuning_required: bool


structured_llm = openai_model.with_structured_output(
    MLPlanningOutput, method="function_calling"
)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a senior ML engineer planning a training run.
            You ONLY use dataset metadata and tool outputs. You never access raw data.
            Keep outputs machine-oriented and concise.
            """,
        ),
        (
            "human",
            """
            Dataset metadata summary:
            {dataset_summary}

            Problem type: {problem_type}
            Allowed evaluation metrics: {evaluation_metrics}
            Class balance summary: {class_balance}

            You MUST return machine-oriented outputs.

            selected_models must ONLY contain:
            - LogisticRegression
            - RandomForestClassifier
            - XGBoostClassifier
            - SVC
            - KNeighborsClassifier

            imbalance_strategy must ONLY be one of:
            - none
            - class_weight
            - undersample
            - smote

            primary_metric must ONLY be one of:
            - accuracy
            - precision
            - recall
            - f1_score
            - roc_auc

            Do NOT include human explanations outside imbalance_reason.
            """,
        ),
    ]
)


ml_planning_chain = prompt | structured_llm
