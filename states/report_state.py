from typing import TypedDict, Any, List, Dict, Tuple
from typing_extensions import NotRequired


class ReportState(TypedDict):

    insights: list[str]

    recommendations: list[str]

    report_path: NotRequired[str]

    html_report_path: NotRequired[str]

    visualization_assets: NotRequired[Dict[str, str]]

    ml_summary: NotRequired[str]

    model_comparison: NotRequired[list[str]]

    eda_narrative: NotRequired[str]

    hypothesis_list: NotRequired[list[str]]

    statistical_tests: NotRequired[Dict[str, Any]]

    data_quality_assessment: NotRequired[Dict[str, Any]]

    business_insights: NotRequired[list[str]]

    visualization_narrative: NotRequired[list[str]]
