from typing import TypedDict, Any, List, Dict, Tuple
from typing_extensions import NotRequired


class DataState(TypedDict):

    file_path: str

    clean_dataset_path: str

    file_type: str

    dtypes: Dict[Any, Any]

    shape: Tuple

    columns: list[str]

    top_rows: List[Dict[Any, Any]]

    bottom_rows: List[Dict[Any, Any]]

    dataset_summary: str

    business_domain: NotRequired[str]

    recommended_tasks: NotRequired[list[str]]

    selected_models: NotRequired[list[str]]

    evaluation_metrics: NotRequired[list[str]]

    encoding_strategy: NotRequired[list[Dict[str, str]]]

    visualization_plan: NotRequired[list[str]]

    scaling_required: NotRequired[bool]

    raw_missing_summary: Dict[Any, Any]

    clean_missing_summary: Dict[Any, Any]

    statistical_summary: Dict[str, Any]

    probable_target_column: str

    categorical_columns: list[str]

    numerical_columns: list[str]

    problem_type: str
