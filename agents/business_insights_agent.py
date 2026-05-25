from __future__ import annotations

from typing import List

from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate

from .models import openai_model


class BusinessInsightsOutput(BaseModel):
    insights: List[str]


structured_llm = openai_model.with_structured_output(
    BusinessInsightsOutput, method="function_calling"
)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an analytical business insights assistant.
            Turn evidence into actionable, cautious insights without causal claims.
            """,
        ),
        (
            "human",
            """
            EDA narrative:
            {eda_narrative}

            Statistical tests:
            {statistical_tests}

            Class balance:
            {class_balance}

            Risk metrics:
            {risk_metrics}

            Generate 3-6 actionable insights with practical implications.
            """,
        ),
    ]
)


business_insights_chain = prompt | structured_llm
