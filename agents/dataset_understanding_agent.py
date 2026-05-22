from .models import openai_model

from langchain_core.prompts import ChatPromptTemplate

from pydantic import BaseModel
from typing import List


class DataUnderstandingOutput(BaseModel):

    dataset_summary: str

    business_domain: str

    probable_target_column: str

    problem_type: str

    recommended_tasks: List[str]

    visualization_recommendations: List[str]

    ml_recommendations: List[str]


structured_llm = (
    openai_model.with_structured_output(
        DataUnderstandingOutput
    )
)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an expert Senior Data Scientist...
            """
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
            """
        )
    ]
)


data_understanding_chain = (
    prompt
    | structured_llm
)