from agents.dataset_understanding_agent import (
    data_understanding_chain,
    DataUnderstandingOutput,
)

from typing import cast

from states.root_state import RootState


async def data_understanding_node(state: RootState):

    data_state = state["data"]

    result = cast(
        DataUnderstandingOutput,
        await data_understanding_chain.ainvoke(
            {
                "shape": data_state["shape"],
                "columns": data_state["columns"],
                "dtypes": data_state["dtypes"],
                "top_rows": data_state["top_rows"],
                "bottom_rows": data_state["bottom_rows"],
                "missing_values": data_state["raw_missing_summary"],
                "statistical_summary": data_state["statistical_summary"],
            }
        ),
    )

    # Convert pydantic output to dict
    result_dict = result.model_dump()

    # Merge into existing data state
    updated_data_state = {
        **data_state,
        **result_dict,
    }

    # Return ONLY the updated section
    return {"data": updated_data_state}
