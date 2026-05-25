from __future__ import annotations

from typing import List

from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate

from .models import openai_model


class ValidationOutput(BaseModel):
    leakage_risk: bool
    leakage_reasons: List[str]
    leakage_recommendations: List[str]
    evidence_summary: List[str] = []
    validation_checks: dict = {}


structured_llm = openai_model.with_structured_output(
    ValidationOutput, method="function_calling"
)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an ML validation agent focused on leakage detection.
            You ONLY use tool outputs and metadata; never access raw data.
            If metrics are perfect or near-perfect, flag leakage risk.
            Return concise, machine-oriented outputs backed by evidence.
            """,
        ),
        (
            "human",
            """
            Experiments:
            {experiments}

            Best model:
            {best_model}

            Metrics summary:
            {best_model_metrics}

            CV stability:
            {cv_stability}

            Evaluation evidence:
            {evaluation_evidence}

            Class balance stats:
            {class_balance}

            You MUST return machine-oriented outputs.

            Use evaluation evidence to justify any leakage risk decision.
            Provide a short evidence_summary list with explicit observations.
            If perfect metrics detected (accuracy, f1_score, or roc_auc == 1.0),
            include evidence-based reasons such as:
            - data leakage
            - target leakage
            - duplicate leakage
            - overfitting
            - train/test contamination

            Include robust evaluation recommendations and explicit checks for:
            - cross-validation stability
            - duplicate overlap
            - suspicious feature-target correlations
            - preprocessing contamination
            """,
        ),
    ]
)


validation_chain = prompt | structured_llm
