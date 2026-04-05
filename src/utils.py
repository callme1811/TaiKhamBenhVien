from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st
from sklearn.preprocessing import OneHotEncoder

GLOBAL_STYLE = """
<style>
html, body, [class*="css"] {
    font-family: "Segoe UI", sans-serif;
}
.block-container {
    max-width: 1380px;
    padding-top: 1rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] * {
    color: white !important;
}
.hero-box {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #1d4ed8 100%);
    border-radius: 24px;
    padding: 28px 32px;
    box-shadow: 0 12px 28px rgba(0,0,0,0.22);
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 18px;
}
.hero-title {
    font-size: 36px;
    font-weight: 800;
    line-height: 1.2;
    color: white;
    margin-bottom: 8px;
}
.hero-subtitle {
    font-size: 15px;
    color: #dbeafe;
    line-height: 1.6;
}
.section-title {
    font-size: 28px;
    font-weight: 800;
    margin: 10px 0 16px 0;
}
.card {
    background: white;
    border: 1px solid rgba(15,23,42,0.06);
    border-radius: 18px;
    padding: 18px 20px;
    margin-bottom: 16px;
    box-shadow: 0 8px 18px rgba(0,0,0,0.06);
}
.metric-card {
    background: linear-gradient(180deg, #ffffff, #f8fafc);
    border: 1px solid rgba(15,23,42,0.08);
    border-radius: 18px;
    padding: 16px 14px;
    text-align: center;
    box-shadow: 0 8px 18px rgba(0,0,0,0.06);
}
.metric-label {
    font-size: 14px;
    color: #475569;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 26px;
    font-weight: 800;
    color: #0f172a;
}
.small-muted {
    color: #94a3b8;
    font-size: 14px;
}
.info-chip {
    display: inline-block;
    background: rgba(37,99,235,0.10);
    color: #1d4ed8;
    border: 1px solid rgba(59,130,246,0.20);
    padding: 6px 12px;
    border-radius: 999px;
    font-size: 13px;
    margin-right: 8px;
    margin-bottom: 8px;
}
div.stButton > button {
    width: 100%;
    border-radius: 14px;
    border: none;
    padding: 0.75rem 1rem;
    font-size: 16px;
    font-weight: 700;
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white;
    box-shadow: 0 10px 20px rgba(37,99,235,0.20);
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #1d4ed8, #1e40af);
    color: white;
}
[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
}
div[data-testid="stMetric"] {
    background: transparent;
    border: none;
    box-shadow: none;
}
</style>
"""

HERO_HTML = """
<div class="hero-box">
    <div class="hero-title">Phân loại nguy cơ tái nhập viện sớm của bệnh nhân đái tháo đường bằng Naive Bayes</div>
    <div class="hero-subtitle">
        Ứng dụng hỗ trợ khám phá dữ liệu, dự đoán nguy cơ tái nhập viện sớm và đánh giá hiệu năng mô hình
        trên hồ sơ bệnh án.
    </div>
</div>
"""


def apply_global_style() -> None:
    st.markdown(GLOBAL_STYLE, unsafe_allow_html=True)
    st.markdown(HERO_HTML, unsafe_allow_html=True)


def add_bar_labels(ax: Any, fmt: str = "{:.0f}") -> None:
    for patch in ax.patches:
        height = patch.get_height()
        ax.annotate(
            fmt.format(height),
            (patch.get_x() + patch.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=10,
        )


def safe_mode(series: pd.Series):
    mode_values = series.mode(dropna=True)
    if len(mode_values) > 0:
        return mode_values.iloc[0]
    return None


def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)
