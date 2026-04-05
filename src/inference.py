from __future__ import annotations

import pandas as pd

from src.preprocessing import add_feature_engineering, clip_input_by_train_range
from src.trainer_utils import format_prediction


def _dense_if_needed(matrix):
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return matrix


def predict_from_input(model, preprocessor, input_data: dict, x_train, numeric_cols, threshold: float = 0.5):
    input_df = pd.DataFrame([input_data])
    input_df = add_feature_engineering(input_df)
    input_df = clip_input_by_train_range(input_df, x_train, numeric_cols)
    x_input = _dense_if_needed(preprocessor.transform(input_df))
    prob = float(model.predict_proba(x_input)[0][1])
    pred, pred_label, display_label = format_prediction(prob, threshold)
    return {
        "input_df": input_df,
        "prob": prob,
        "pred": pred,
        "pred_label": pred_label,
        "display_label": display_label,
    }
