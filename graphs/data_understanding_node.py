from agents.dataset_understanding_agent import (
    data_understanding_chain, DataUnderstandingOutput
)
from typing import cast

from state import GraphState


async def data_understanding_node(
    state: GraphState
):

    result = cast(DataUnderstandingOutput, await data_understanding_chain.ainvoke(
        {
            "shape": state["shape"],
            "columns": state["columns"],
            "dtypes": state["dtypes"],
            "top_rows": state["top_rows"],
            "bottom_rows": state["bottom_rows"],
            "missing_values":
                state["raw_missing_summary"],
            "statistical_summary":
                state["statistical_summary"]
        }
    ))

    return result.model_dump()