from typing import TypedDict, Any, List, Dict, Tuple


class MLState(TypedDict):

    selected_models: list[str]

    experiments: list[dict]

    best_model_path: str

    best_model_metrics: dict
