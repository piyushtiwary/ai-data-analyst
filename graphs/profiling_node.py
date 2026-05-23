from states.root_state import RootState

import pandas as pd

from typing import cast, List, Dict, Any


async def profiling_node(state: RootState):

    data_state = state["data"]

    df = pd.read_csv(data_state["clean_dataset_path"])

    return {
        "data": {
            **data_state,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "shape": df.shape,
            "top_rows": cast(
                List[Dict[str, Any]],
                df.head(5).to_dict(orient="records"),
            ),
            "bottom_rows": cast(
                List[Dict[str, Any]],
                df.tail(5).to_dict(orient="records"),
            ),
            "clean_missing_summary": df.isnull().sum().to_dict(),
            "statistical_summary": cast(
                Dict[str, Any],
                df.describe(include="all").fillna("N/A").to_dict(),
            ),
        }
    }
