from typing import TypedDict, Any, List, Dict, Tuple


class ReportState(TypedDict):

    insights: list[str]

    recommendations: list[str]

    report_path: str
