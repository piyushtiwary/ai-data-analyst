from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate

from .models import openai_model


class EDANarrativeOutput(BaseModel):
    eda_narrative: str
    hypotheses: List[str]


structured_llm = openai_model.with_structured_output(
    EDANarrativeOutput, method="function_calling"
)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a data scientist focused on exploratory analysis and statistical reasoning.
            You ONLY use metadata and tool outputs; never use raw data.
            Write nuanced, cautious interpretations with uncertainty and limitations.
            """,
        ),
        (
            "human",
            """
            Dataset summary:
            {dataset_summary}

            Statistical summary:
            {statistical_summary}

            Class balance:
            {class_balance}

            Feature-target correlations (top):
            {feature_target_correlations}

            Statistical tests:
            {statistical_tests}

            Data quality signals:
            {data_quality}

            Generate a cohesive EDA narrative that explains patterns,
            cautions against over-interpretation, and highlights uncertainty.
            Also propose 3-6 testable hypotheses (no causal claims).
            """,
        ),
    ]
)


eda_reasoning_chain = prompt | structured_llm
