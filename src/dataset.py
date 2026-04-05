from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from src.config import DATA_PATH
from src.preprocessing import clean_raw_data
from src.utils import safe_mode


@st.cache_data
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"Không tìm thấy file dữ liệu: {path}")
        st.stop()
    df = pd.read_csv(path)
    return clean_raw_data(df)


def build_default_values(x_train: pd.DataFrame, numeric_cols, categorical_cols):
    defaults = {}
    for col in numeric_cols:
        series = pd.to_numeric(x_train[col], errors="coerce").dropna()
        defaults[col] = float(series.median()) if len(series) > 0 else 0.0
    for col in categorical_cols:
        mode_val = safe_mode(x_train[col].astype(str))
        defaults[col] = str(mode_val) if mode_val is not None else "Unknown"
    return defaults


def build_input_limits(x_train: pd.DataFrame, numeric_cols):
    limits = {}
    for col in numeric_cols:
        series = pd.to_numeric(x_train[col], errors="coerce").dropna()
        if len(series) > 0:
            q01 = float(series.quantile(0.01))
            q99 = float(series.quantile(0.99))
            q50 = float(series.median())
            if q01 == q99:
                q01 = float(series.min())
                q99 = float(series.max())
            limits[col] = {"min": q01, "max": q99, "default": q50}
        else:
            limits[col] = {"min": 0.0, "max": 100.0, "default": 0.0}
    return limits
