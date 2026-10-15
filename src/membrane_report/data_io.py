"""Small readers for the project's own tables."""

from __future__ import annotations

import pandas as pd


def read_selection_log(path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip", dtype={"qualifies": str, "selected": str})
