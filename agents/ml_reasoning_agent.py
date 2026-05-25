from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from .models import openai_model


class RecommendedTuning(BaseModel):
    model: str = Field(default="")
    params: Dict[str, List[Any]] = Field(default_factory=dict)


class MLReasoningOutput(BaseModel):
    next_action: str
    tuning_required: bool
    recommended_tuning: RecommendedTuning


structured_llm = openai_model.with_structured_output(
    MLReasoningOutput, method="function_calling"
)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a senior ML engineer that decides next steps after training.
            You ONLY use tool outputs and metadata; never access raw data.
            Return concise, machine-oriented outputs that align with validation risk.
            """,
        ),
        (
            "human",
            """
            Experiments:
            {experiments}

            Class balance stats:
            {class_balance}

            Best model:
            {best_model}

            Metrics summary:
            {best_model_metrics}

            Feature importance (top features):
            {feature_importance}

            Leakage risk:
            {leakage_risk}

            You MUST return machine-oriented outputs.

            If leakage_risk is true, avoid deployment recommendations.
            If tuning is required, suggest a compact parameter grid.
            If tuning is not required, keep recommended_tuning empty.
            """,
        ),
    ]
)


ml_reasoning_chain = prompt | structured_llm
