from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from src.utils import make_ohe


def normalize_target_value(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        if np.isnan(value):
            return np.nan
        if value in (0.0, 1.0):
            return int(value)

    text = str(value).strip().lower()
    mapping = {
        "<30": 1,
        "yes": 1,
        "y": 1,
        "true": 1,
        "1": 1,
        "readmitted": 1,
        "positive": 1,
        "high": 1,
        "risk": 1,
        ">30": 0,
        "no": 0,
        "n": 0,
        "false": 0,
        "0": 0,
        "not readmitted": 0,
        "negative": 0,
        "low": 0,
        "none": 0,
    }
    if text in mapping:
        return mapping[text]
    try:
        num = float(text)
        if num in (0.0, 1.0):
            return int(num)
    except Exception:
        pass
    return np.nan


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data = data.replace("?", np.nan)
    data = data.replace("None", np.nan)
    data = data.replace("", np.nan)

    rename_map = {
        "A1Ctest": "A1Cresult",
        "glucose_test": "max_glu_serum",
        "num_lab_procedures": "n_lab_procedures",
        "num_procedures": "n_procedures",
        "num_medications": "n_medications",
        "number_outpatient": "n_outpatient",
        "number_inpatient": "n_inpatient",
        "number_emergency": "n_emergency",
    }
    available_rename = {k: v for k, v in rename_map.items() if k in data.columns and v not in data.columns}
    if available_rename:
        data = data.rename(columns=available_rename)

    if "readmitted" in data.columns:
        data["readmitted"] = data["readmitted"].apply(normalize_target_value)
    return data


def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    numeric_candidates = [
        "time_in_hospital",
        "n_lab_procedures",
        "n_procedures",
        "n_medications",
        "n_outpatient",
        "n_inpatient",
        "n_emergency",
        "number_diagnoses",
    ]
    for col in numeric_candidates:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    if all(col in data.columns for col in ["n_outpatient", "n_inpatient", "n_emergency"]):
        data["total_visits"] = (
            data["n_outpatient"].fillna(0)
            + data["n_inpatient"].fillna(0)
            + data["n_emergency"].fillna(0)
        )
    if all(col in data.columns for col in ["n_inpatient", "n_outpatient"]):
        data["visit_ratio"] = data["n_inpatient"].fillna(0) / (data["n_outpatient"].fillna(0) + 1)
    if all(col in data.columns for col in ["time_in_hospital", "n_lab_procedures"]):
        data["severity"] = data["time_in_hospital"].fillna(0) * data["n_lab_procedures"].fillna(0)
    if all(col in data.columns for col in ["n_medications", "n_procedures"]):
        data["med_ratio"] = data["n_medications"].fillna(0) / (data["n_procedures"].fillna(0) + 1)
    if all(col in data.columns for col in ["n_medications", "time_in_hospital"]):
        data["care_intensity"] = data["n_medications"].fillna(0) / (data["time_in_hospital"].fillna(0) + 1)
    if "age" in data.columns:
        age_str = data["age"].astype(str)
        data["is_elderly"] = age_str.str.contains("70|80|90", regex=True, na=False).astype(int)
    if "time_in_hospital" in data.columns:
        data["long_stay"] = (data["time_in_hospital"].fillna(0) > 7).astype(int)
    if "n_medications" in data.columns:
        data["high_med"] = (data["n_medications"].fillna(0) > 10).astype(int)
    if all(col in data.columns for col in ["n_lab_procedures", "n_procedures"]):
        data["total_procedures"] = data["n_lab_procedures"].fillna(0) + data["n_procedures"].fillna(0)
    return data


def prepare_xy(df: pd.DataFrame):
    data = add_feature_engineering(df)
    if "readmitted" not in data.columns:
        raise ValueError("Dữ liệu không có cột mục tiêu 'readmitted'.")

    data["readmitted"] = data["readmitted"].apply(normalize_target_value)
    data = data.dropna(subset=["readmitted"]).copy()
    data["readmitted"] = data["readmitted"].astype(int)

    X = data.drop(columns=["readmitted"])
    y = data["readmitted"]

    categorical_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    return X, y, categorical_cols, numeric_cols, data


def build_preprocessor(numeric_cols, categorical_cols) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_ohe()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )


def clip_input_by_train_range(input_df: pd.DataFrame, x_train: pd.DataFrame, numeric_cols):
    data = input_df.copy()
    for col in numeric_cols:
        if col in data.columns and col in x_train.columns:
            train_col = pd.to_numeric(x_train[col], errors="coerce").dropna()
            if len(train_col) > 0:
                low = float(train_col.quantile(0.01))
                high = float(train_col.quantile(0.99))
                data[col] = pd.to_numeric(data[col], errors="coerce").clip(lower=low, upper=high)
    return data
