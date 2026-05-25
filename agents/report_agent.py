from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from .models import openai_model


class ReportOutput(BaseModel):
    title: str
    html_content: str


structured_llm = openai_model.with_structured_output(
    ReportOutput, method="function_calling"
)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an ML analyst and research assistant.
            You ONLY use metadata and tool outputs. Never use raw data or model internals.
            Write cohesive, human-readable sections with clear explanations and narrative flow.
            The report should read like an AutoML platform analyst: evidence-driven
            and focused on data understanding first.
                Use analytical, narrative language similar to a research assistant.
                Avoid statements like "caution is warranted" or "overfitting" unless explicitly
                supported by validation checks or statistical tests.
            """,
        ),
        (
            "human",
            """
            Dataset summary:
            {dataset_summary}

            Business domain:
            {business_domain}

            Shape: {shape}
            Columns: {columns}
            Dtypes: {dtypes}
            Target: {target_column}
            Problem type: {problem_type}

            Missing values (raw):
            {raw_missing_summary}

            Missing values (clean):
            {clean_missing_summary}

            Statistical summary (condensed):
            {statistical_summary}

            Encoding strategy:
            {encoding_strategy}

            Scaling required:
            {scaling_required}

            Class balance stats:
            {class_balance}

            Recommended tasks:
            {recommended_tasks}

            ML summary:
            {ml_summary}

            Performance metrics:
            {performance_metrics}

            Stability metrics:
            {stability_metrics}

            Risk metrics:
            {risk_metrics}

            Business metrics:
            {business_metrics}

            Model comparison:
            {model_comparison}

            Threshold optimization:
            {threshold_optimization}

            Validation flags:
            {validation_flags}

            Evaluation evidence:
            {evaluation_evidence}

            EDA narrative:
            {eda_narrative}

            Hypotheses:
            {hypotheses}

            Statistical testing results:
            {statistical_tests}

            Data quality assessment:
            {data_quality_assessment}

            Business insights:
            {business_insights}

            Visualization narrative:
            {visualization_narrative}

            Visualization assets:
            {visualization_assets}

            SHAP summary:
            {shap_summary}
            You MUST return machine-oriented outputs.

            Produce a single HTML document in html_content with these sections in order:
            0) Key Findings (card layout)
            1) Executive Summary
            2) Dataset Overview
            3) Data Quality and Missingness
            4) Target and Class Balance
            5) Feature Summary and Distributions
            6) Modeling Approach and Evaluation
            7) Threshold Optimization and Decision Guidance
            8) Leakage and Validation Risks
            9) Limitations and Constraints
            10) Visualizations and Interpretation

            Use semantic tags: <main>, <section>, <h1>, <h2>, <p>, <ul>, <li>, <figure>, <figcaption>.
            The Key Findings section must be a <section class="key-findings"> containing a
            <div class="cards"> with 3-6 <div class="card"> items. Each card should have
            a <div class="card-title"> and a concise <p>.
            Include evidence-based items such as class imbalance, dominant predictors,
            ROC-AUC level, and any validation flags.
            For visualizations, include <img src="..."> using provided image paths.
            All visualization sections MUST include a plain-language explanation of what the plot shows.
            Use the SHAP summary to explain model drivers in plain language.
            Use the evidence to explain WHY conclusions are reached.
            Include a Statistical Interpretation paragraph that references statistical tests,
            describing statistical vs practical significance in plain language.
            Include a dedicated Limitations and Constraints section with a short <ul> that
            covers: severe class imbalance, limited demographic diversity, cross-sectional
            dataset, potential leakage concerns, and limited sample size.
            Prioritize interpretability and uncertainty analysis over model optimization.
            Keep sections cohesive and readable; use full sentences.
            """,
        ),
    ]
)


report_chain = prompt | structured_llm
