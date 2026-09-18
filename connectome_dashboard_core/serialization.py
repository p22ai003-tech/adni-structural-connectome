from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [json_safe(item) for item in value]
    if pd.isna(value):
        return None
    return str(value)


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {str(key): json_safe(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]
