from .models import openai_model

from langchain_core.prompts import ChatPromptTemplate

from pydantic import BaseModel
from typing import List, Dict


class EncodingRule(BaseModel):

    column_name: str

    encoding_type: str


class DataUnderstandingOutput(BaseModel):

    dataset_summary: str

    business_domain: str

    probable_target_column: str

    categorical_columns: List[str]

    numerical_columns: List[str]

    problem_type: str

    recommended_tasks: List[str]

    selected_models: List[str]

    evaluation_metrics: List[str]

    encoding_strategy: List[EncodingRule]

    visualization_plan: List[str]

    scaling_required: bool


structured_llm = openai_model.with_structured_output(
    DataUnderstandingOutput, method="function_calling"
)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an expert Senior Data Scientist...
            """,
        ),
        (
            "human",
            """
            Analyze the following dataset metadata.

            DATASET SHAPE:
            {shape}

            COLUMNS:
            {columns}

            DATATYPES:
            {dtypes}

            TOP ROWS:
            {top_rows}

            BOTTOM ROWS:
            {bottom_rows}

            MISSING VALUES:
            {missing_values}

            STATISTICAL SUMMARY:
            {statistical_summary}
            
            You MUST return machine-oriented outputs.

            selected_models must ONLY contain:
            - LogisticRegression
            - RandomForestClassifier
            - XGBoostClassifier
            - SVC
            - KNeighborsClassifier

            evaluation_metrics must ONLY contain:
            - accuracy
            - precision
            - recall
            - f1_score
            - roc_auc

            visualization_plan must ONLY contain:
            - confusion_matrix
            - roc_curve
            - feature_importance
            - correlation_heatmap
            - pairplot
            - class_distribution

            encoding_strategy values must ONLY contain:
            - one_hot
            - ordinal
            - label_encoding

            Do NOT generate human explanations inside these fields.
            These fields are intended for downstream automated execution.
            """,
        ),
    ]
)


data_understanding_chain = prompt | structured_llm
