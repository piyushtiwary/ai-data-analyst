from __future__ import annotations

from typing import List

from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate

from .models import openai_model


class HypothesisOutput(BaseModel):
    hypotheses: List[str]


structured_llm = openai_model.with_structured_output(
    HypothesisOutput, method="function_calling"
)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You generate plausible, testable hypotheses based on metadata and analysis outputs.
            Avoid claims of causality; keep hypotheses actionable.
            """,
        ),
        (
            "human",
            """
            Dataset summary:
            {dataset_summary}

            Class balance:
            {class_balance}

            Feature-target correlations (top):
            {feature_target_correlations}

            Statistical tests:
            {statistical_tests}

            Generate 3-6 potential hypotheses for further testing.
            """,
        ),
    ]
)


hypothesis_chain = prompt | structured_llm
