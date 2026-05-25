from __future__ import annotations

from typing import List

from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate

from .models import openai_model


class VisualizationNarrativeOutput(BaseModel):
    narratives: List[str]


structured_llm = openai_model.with_structured_output(
    VisualizationNarrativeOutput, method="function_calling"
)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You explain charts in plain language with analytical context.
            Use only metadata and tool outputs.
            """,
        ),
        (
            "human",
            """
            Visualization assets:
            {visualization_assets}

            Class balance:
            {class_balance}

            Performance metrics:
            {performance_metrics}

            Stability metrics:
            {stability_metrics}

            Validation flags:
            {validation_flags}

            SHAP summary:
            {shap_summary}

            Provide 1-2 sentences per visualization with insight-driven interpretation.
            Tie interpretations to evidence where available.
            Explain WHY key features matter in context (e.g., sleep duration, anxiety level,
            and stress level as strong contributors aligned with mental health research).
            """,
        ),
    ]
)


visualization_narrative_chain = prompt | structured_llm
