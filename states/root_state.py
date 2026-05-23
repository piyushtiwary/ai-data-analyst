from typing import TypedDict, Any, List, Dict, Tuple

from .data_state import DataState
from .ml_state import MLState
from .report_state import ReportState


class RootState(TypedDict):

    data: DataState

    ml: MLState

    report: ReportState
