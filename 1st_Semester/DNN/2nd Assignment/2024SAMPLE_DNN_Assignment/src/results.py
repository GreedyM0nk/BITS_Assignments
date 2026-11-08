from pathlib import Path
from typing import Any

import pandas as pd

from .config import RESULT_COLUMNS


def append_result(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([{column: row.get(column, "") for column in RESULT_COLUMNS}])
    if path.exists():
        frame.to_csv(path, mode="a", header=False, index=False)
    else:
        frame.to_csv(path, index=False)


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=RESULT_COLUMNS)
    frame = pd.read_csv(path)
    missing = [c for c in RESULT_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"results.csv is missing required columns: {missing}")
    return frame[RESULT_COLUMNS]
