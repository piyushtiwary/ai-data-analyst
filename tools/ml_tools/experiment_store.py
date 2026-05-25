from __future__ import annotations

from typing import Any, Dict, List

import json
import os

from langchain.tools import tool


@tool
def append_experiment_tool(
    experiment: Dict[str, Any], store_path: str = "outputs/experiment_history.jsonl"
) -> Dict[str, Any]:
    """
    Append a single experiment record to a JSONL store for cross-run memory.
    """
    os.makedirs(os.path.dirname(store_path), exist_ok=True)

    with open(store_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(experiment) + "\n")

    return {"store_path": store_path}


@tool
def load_experiment_history_tool(
    store_path: str = "outputs/experiment_history.jsonl",
) -> Dict[str, Any]:
    """
    Load experiment history from a JSONL store and return all records.
    """
    if not os.path.exists(store_path):
        return {"experiments": []}

    experiments: List[Dict[str, Any]] = []
    with open(store_path, "r", encoding="utf-8") as handle:
        for line in handle:
            try:
                experiments.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return {"experiments": experiments}
