from __future__ import annotations

import os

import joblib
import streamlit as st
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from src.config import (
    CALIBRATION_CV,
    MODEL_DIR,
    MODEL_PATH,
    NB_ALPHA,
    PREPROCESSOR_PATH,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.preprocessing import build_preprocessor, prepare_xy


def _dense_if_needed(matrix):
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return matrix


@st.cache_resource(show_spinner=False)
def prepare_and_train_model(df):
    x, y, categorical_cols, numeric_cols, _ = prepare_xy(df)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    x_train_processed = _dense_if_needed(preprocessor.fit_transform(x_train))

    base_model = MultinomialNB(alpha=NB_ALPHA)
    model = CalibratedClassifierCV(base_model, method="sigmoid", cv=CALIBRATION_CV)
    model.fit(x_train_processed, y_train)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    return model, preprocessor, x_train, x_test, y_train, y_test, categorical_cols, numeric_cols


@st.cache_resource(show_spinner=False)
def load_or_train_model(df):
    x, y, categorical_cols, numeric_cols, _ = prepare_xy(df)

    if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            preprocessor = joblib.load(PREPROCESSOR_PATH)
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                stratify=y,
            )
            return model, preprocessor, x_train, x_test, y_train, y_test, categorical_cols, numeric_cols
        except Exception:
            return prepare_and_train_model(df)

    return prepare_and_train_model(df)
