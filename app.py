
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    classification_report,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV


# =========================================================
# CẤU HÌNH TRANG
# =========================================================
st.set_page_config(
    page_title="Phân loại nguy cơ tái nhập viện sớm",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero-box">
    <div class="hero-title">Phân loại nguy cơ tái nhập viện sớm của bệnh nhân đái tháo đường bằng Naive Bayes</div>
    <div class="hero-subtitle">
        Ứng dụng hỗ trợ khám phá dữ liệu, dự đoán nguy cơ tái nhập viện sớm và đánh giá hiệu năng mô hình
        trên hồ sơ bệnh án.
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# =========================================================
# HẰNG SỐ
# =========================================================
DATA_PATH = "hospital_readmissions.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "multinomial_nb_calibrated_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor_nb.pkl")


# =========================================================
# HÀM PHỤ
# =========================================================
def add_bar_labels(ax, fmt="{:.0f}"):
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            fmt.format(height),
            (p.get_x() + p.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=10,
        )


def safe_mode(series: pd.Series):
    mode_values = series.mode(dropna=True)
    if len(mode_values) > 0:
        return mode_values.iloc[0]
    return None


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


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

    # fallback: nếu là chuỗi số "0.0"/"1.0"
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

    # Chuẩn hóa vài tên cột hay gặp
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


def clip_input_by_train_range(input_df: pd.DataFrame, X_train: pd.DataFrame, numeric_cols):
    data = input_df.copy()

    for col in numeric_cols:
        if col in data.columns and col in X_train.columns:
            train_col = pd.to_numeric(X_train[col], errors="coerce").dropna()
            if len(train_col) > 0:
                low = float(train_col.quantile(0.01))
                high = float(train_col.quantile(0.99))
                data[col] = pd.to_numeric(data[col], errors="coerce").clip(lower=low, upper=high)

    return data


@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"Không tìm thấy file dữ liệu: {DATA_PATH}")
        st.stop()

    df = pd.read_csv(DATA_PATH)
    df = clean_raw_data(df)
    return df


def build_default_values(X_train: pd.DataFrame, numeric_cols, categorical_cols):
    defaults = {}

    for col in numeric_cols:
        series = pd.to_numeric(X_train[col], errors="coerce").dropna()
        defaults[col] = float(series.median()) if len(series) > 0 else 0.0

    for col in categorical_cols:
        mode_val = safe_mode(X_train[col].astype(str))
        defaults[col] = str(mode_val) if mode_val is not None else "Unknown"

    return defaults


def build_input_limits(X_train: pd.DataFrame, numeric_cols):
    limits = {}

    for col in numeric_cols:
        series = pd.to_numeric(X_train[col], errors="coerce").dropna()
        if len(series) > 0:
            q01 = float(series.quantile(0.01))
            q99 = float(series.quantile(0.99))
            q50 = float(series.median())

            if q01 == q99:
                q01 = float(series.min())
                q99 = float(series.max())

            limits[col] = {
                "min": q01,
                "max": q99,
                "default": q50,
            }
        else:
            limits[col] = {
                "min": 0.0,
                "max": 100.0,
                "default": 0.0,
            }

    return limits


def prepare_xy(df: pd.DataFrame):
    data = add_feature_engineering(df)

    if "readmitted" not in data.columns:
        raise ValueError("Dữ liệu không có cột mục tiêu 'readmitted'.")

    data["readmitted"] = data["readmitted"].apply(normalize_target_value)
    data = data.dropna(subset=["readmitted"]).copy()
    data["readmitted"] = data["readmitted"].astype(int)

    target_col = "readmitted"
    X = data.drop(columns=[target_col])
    y = data[target_col]

    categorical_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    return X, y, categorical_cols, numeric_cols, data


def build_preprocessor(numeric_cols, categorical_cols):
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


@st.cache_resource(show_spinner=False)
def prepare_and_train_model(df: pd.DataFrame):
    X, y, categorical_cols, numeric_cols, _ = prepare_xy(df)

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    if hasattr(X_train_processed, "toarray"):
        X_train_processed = X_train_processed.toarray()
    if hasattr(X_test_processed, "toarray"):
        X_test_processed = X_test_processed.toarray()

    base_model = MultinomialNB(alpha=1.0)
    model = CalibratedClassifierCV(base_model, method="sigmoid", cv=3)
    model.fit(X_train_processed, y_train)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    return model, preprocessor, X_train, X_test, y_train, y_test, categorical_cols, numeric_cols


@st.cache_resource(show_spinner=False)
def load_or_train_model(df: pd.DataFrame):
    X, y, categorical_cols, numeric_cols, _ = prepare_xy(df)

    if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            preprocessor = joblib.load(PREPROCESSOR_PATH)

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y,
            )

            return model, preprocessor, X_train, X_test, y_train, y_test, categorical_cols, numeric_cols
        except Exception:
            return prepare_and_train_model(df)

    return prepare_and_train_model(df)


def evaluate_model(model, preprocessor, X_test, y_test, threshold=0.5):
    X_test_processed = preprocessor.transform(X_test)

    if hasattr(X_test_processed, "toarray"):
        X_test_processed = X_test_processed.toarray()

    y_prob = model.predict_proba(X_test_processed)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    result = {
        "y_prob": y_prob,
        "y_pred": y_pred,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_pos": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        "recall_pos": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "f1_pos": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }

    try:
        result["auc"] = roc_auc_score(y_test, y_prob)
    except Exception:
        result["auc"] = 0.0

    return result


# =========================================================
# LOAD DỮ LIỆU + MODEL
# =========================================================
df = load_data()

(
    model,
    preprocessor,
    X_train,
    X_test,
    y_train,
    y_test,
    categorical_cols,
    numeric_cols,
) = load_or_train_model(df)

default_values = build_default_values(X_train, numeric_cols, categorical_cols)
input_limits = build_input_limits(X_train, numeric_cols)

st.sidebar.markdown("## Điều hướng")
st.sidebar.markdown("<div class='small-muted'>Chọn nội dung muốn xem</div>", unsafe_allow_html=True)

page = st.sidebar.radio(
    "",
    [
        "Giới thiệu & EDA",
        "Triển khai mô hình",
        "Đánh giá hiệu năng",
        "So sánh & cải tiến",
    ],
)

st.sidebar.markdown("---")
threshold = st.sidebar.slider(
    "Ngưỡng dự đoán",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Thông tin mô hình")
st.sidebar.markdown("**Thuật toán:** Multinomial Naive Bayes + Calibration")
st.sidebar.markdown("**Bài toán:** Phân loại nhị phân")
st.sidebar.markdown("**Mục tiêu:** Dự đoán tái nhập viện sớm")
st.sidebar.markdown("**Mã nhãn:** 1 = tái nhập viện sớm, 0 = không tái nhập viện sớm")
st.sidebar.markdown(f"**Ngưỡng hiện tại:** {threshold:.2f}")
st.sidebar.markdown(f"**Nguồn dữ liệu:** {DATA_PATH}")

eval_result = evaluate_model(model, preprocessor, X_test, y_test, threshold=threshold)
y_prob = eval_result["y_prob"]
y_pred = eval_result["y_pred"]


# =========================================================
# TRANG 1 - GIỚI THIỆU & EDA
# =========================================================
if page == "Giới thiệu & EDA":
    st.markdown("<div class='section-title'>Trang 1: Giới thiệu & Khám phá dữ liệu (EDA)</div>", unsafe_allow_html=True)

    target_dist = df["readmitted"].value_counts(dropna=False).to_dict() if "readmitted" in df.columns else {}

    st.markdown(
        """
        <div class="card">
            <h3>1. Thông tin bài toán</h3>
            <p><b>Tên đề tài:</b> Phân loại nguy cơ tái nhập viện sớm của bệnh nhân đái tháo đường bằng Naive Bayes</p>
            <p><b>Bản chất bài toán:</b> Phân loại nhị phân, trong đó <b>1</b> là bệnh nhân tái nhập viện sớm, <b>0</b> là còn lại.</p>
            <p><b>Giá trị thực tiễn:</b> Mô hình hỗ trợ nhận diện sớm bệnh nhân có nguy cơ quay lại bệnh viện trong thời gian ngắn, giúp bác sĩ theo dõi sát hơn và hỗ trợ phân bổ nguồn lực điều trị hợp lý hơn.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='card'><h3>2. Thông tin dữ liệu</h3></div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"<div class='info-chip'>Số dòng: {df.shape[0]}</div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='info-chip'>Số cột: {df.shape[1]}</div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='info-chip'>Phân bố nhãn: {target_dist}</div>", unsafe_allow_html=True)

    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("<div class='card'><h3>3. Kiểm tra dữ liệu thiếu</h3></div>", unsafe_allow_html=True)
    missing_df = pd.DataFrame({
        "Tên cột": df.columns,
        "Số giá trị thiếu": df.isnull().sum().values,
        "Tỷ lệ thiếu (%)": (df.isnull().sum().values / len(df) * 100).round(2),
    }).sort_values(by="Số giá trị thiếu", ascending=False)
    st.dataframe(missing_df, use_container_width=True)

    st.markdown("<div class='card'><h3>4. Biểu đồ phân tích dữ liệu</h3></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if "readmitted" in df.columns:
            fig1, ax1 = plt.subplots(figsize=(8, 4.5))
            counts = df["readmitted"].fillna(-1).value_counts().sort_index()
            counts.plot(kind="bar", ax=ax1)
            ax1.set_title("Phân bố nhãn readmitted", fontsize=14, fontweight="bold")
            ax1.set_xlabel("Nhãn")
            ax1.set_ylabel("Số lượng")
            ax1.tick_params(axis="x", rotation=0)
            add_bar_labels(ax1)
            plt.tight_layout()
            st.pyplot(fig1)

    with col2:
        if "age" in df.columns:
            fig2, ax2 = plt.subplots(figsize=(8, 4.5))
            age_counts = df["age"].astype(str).value_counts().sort_index()
            age_counts.plot(kind="bar", ax=ax2)
            ax2.set_title("Phân bố nhóm tuổi", fontsize=14, fontweight="bold")
            ax2.set_xlabel("Nhóm tuổi")
            ax2.set_ylabel("Số lượng")
            ax2.tick_params(axis="x", rotation=45)
            plt.tight_layout()
            st.pyplot(fig2)

    if all(col in df.columns for col in ["readmitted", "time_in_hospital"]):
        temp_df = df.copy()
        temp_df["time_in_hospital"] = pd.to_numeric(temp_df["time_in_hospital"], errors="coerce")
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        avg_stay = temp_df.groupby("readmitted")["time_in_hospital"].mean()
        avg_stay.plot(kind="bar", ax=ax3)
        ax3.set_title("Thời gian nằm viện trung bình theo nhãn", fontsize=16, fontweight="bold")
        ax3.set_xlabel("Nhãn readmitted")
        ax3.set_ylabel("Số ngày nằm viện trung bình")
        ax3.tick_params(axis="x", rotation=0)
        add_bar_labels(ax3, fmt="{:.2f}")
        plt.tight_layout()
        st.pyplot(fig3)

    st.markdown(
        """
        <div class="card">
            <h3>5. Nhận xét dữ liệu</h3>
            <ul>
                <li>Dữ liệu gồm cả biến số và biến phân loại.</li>
                <li>Cột mục tiêu là <b>readmitted</b>, đã được quy đổi thành bài toán nhị phân để phát hiện tái nhập viện sớm.</li>
                <li>Một số đặc trưng như thời gian nằm viện, số thuốc, số lần nhập viện và số xét nghiệm có thể liên quan đến nguy cơ tái nhập viện.</li>
                <li>Dữ liệu có giá trị thiếu nên cần bước tiền xử lý trước khi huấn luyện mô hình.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="card">
            <h3>6. Giải thích bài toán</h3>
            <ul>
                <li><b>1</b>: bệnh nhân tái nhập viện sớm.</li>
                <li><b>0</b>: bệnh nhân không tái nhập viện sớm.</li>
                <li>Mục tiêu của mô hình là phát hiện sớm nhóm nguy cơ để ưu tiên theo dõi.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# TRANG 2 - TRIỂN KHAI MÔ HÌNH
# =========================================================
elif page == "Triển khai mô hình":
    st.markdown("<div class='section-title'>Trang 2: Triển khai mô hình</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="card">
            <h3>1. Mô tả quy trình xử lý</h3>
            <p>Dữ liệu đầu vào được làm sạch, tạo thêm đặc trưng mới, xử lý giá trị thiếu, chuẩn hoá về miền không âm, mã hoá biến phân loại, sau đó đưa vào mô hình Multinomial Naive Bayes để dự đoán xác suất tái nhập viện sớm.</p>
            <p><b>Pipeline:</b> Làm sạch dữ liệu → Feature Engineering → Impute → MinMaxScaler + OneHotEncoder → Naive Bayes → Calibration → Dự đoán xác suất → Phân lớp theo threshold</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.25, 0.75])

    selected_input_cols = [
        "age",
        "time_in_hospital",
        "n_lab_procedures",
        "n_procedures",
        "n_medications",
        "n_outpatient",
        "n_inpatient",
        "n_emergency",
        "number_diagnoses",
        "gender",
        "A1Cresult",
        "max_glu_serum",
        "insulin",
        "change",
        "diabetesMed",
    ]

    available_input_cols = [col for col in selected_input_cols if col in X_train.columns]

    with left:
        st.markdown("<div class='card'><h3>2. Nhập dữ liệu bệnh nhân</h3></div>", unsafe_allow_html=True)

        note = st.text_input("Ghi chú bệnh nhân (tuỳ chọn)")
        input_data = {}

        for col in available_input_cols:
            if col in numeric_cols:
                min_val = float(input_limits[col]["min"])
                max_val = float(input_limits[col]["max"])
                default_val = float(input_limits[col]["default"])

                if min_val == max_val:
                    max_val = min_val + 1.0

                input_data[col] = st.number_input(
                    label=col,
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val,
                    step=1.0 if max_val > 5 else 0.1,
                )

            elif col in categorical_cols:
                options = sorted(X_train[col].astype(str).dropna().unique().tolist())
                if not options:
                    options = ["Unknown"]

                default_val = str(default_values.get(col, options[0]))
                default_index = options.index(default_val) if default_val in options else 0

                input_data[col] = st.selectbox(
                    col,
                    options=options,
                    index=default_index,
                )

        for col in X_train.columns:
            if col not in input_data:
                input_data[col] = default_values.get(col, 0 if col in numeric_cols else "Unknown")

    with right:
        st.markdown(
            """
            <div class="card">
                <h3>3. Xử lý kỹ thuật</h3>
                <ul>
                    <li>Làm sạch dữ liệu, thay thế giá trị thiếu.</li>
                    <li>Tạo thêm các đặc trưng như total_visits, severity, care_intensity...</li>
                    <li>Mã hóa biến phân loại bằng OneHotEncoder.</li>
                    <li>Chuẩn hoá đặc trưng số bằng MinMaxScaler để phù hợp với Multinomial Naive Bayes.</li>
                    <li>Hiệu chỉnh xác suất bằng Calibration để kết quả bớt cực đoan.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="card">
                <h3>4. Cấu hình mô hình</h3>
                <p><b>Thuật toán:</b> Multinomial Naive Bayes + Calibration</p>
                <p><b>Ngưỡng dự đoán:</b> {threshold:.2f}</p>
                <p><b>Đầu ra:</b> Xác suất + nhãn phân loại</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="card">
                <h3>5. Ý nghĩa đầu ra</h3>
                <p>Nếu xác suất dự đoán lớn hơn hoặc bằng ngưỡng đang chọn, bệnh nhân sẽ được xếp vào nhóm <b>nguy cơ tái nhập viện sớm</b>.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div class='card'><h3>6. Kết quả dự đoán</h3></div>", unsafe_allow_html=True)

    if st.button("Dự đoán nguy cơ tái nhập viện sớm"):
        input_df = pd.DataFrame([input_data])
        input_df = add_feature_engineering(input_df)
        input_df = clip_input_by_train_range(input_df, X_train, numeric_cols)

        X_input = preprocessor.transform(input_df)
        if hasattr(X_input, "toarray"):
            X_input = X_input.toarray()

        prob = model.predict_proba(X_input)[0][1]
        pred = 1 if prob >= threshold else 0
        pred_label = "YES" if pred == 1 else "NO"

        if note:
            st.info(f"Ghi chú bệnh nhân: {note}")

        c1, c2 = st.columns([1.4, 0.6])

        with c1:
            if pred_label == "YES":
                st.error("Kết quả: Bệnh nhân có nguy cơ tái nhập viện sớm")
            else:
                st.success("Kết quả: Bệnh nhân không có nguy cơ tái nhập viện sớm")

        with c2:
            st.metric("Xác suất nguy cơ", f"{prob:.2%}")

        st.progress(float(prob))

        st.markdown("### Giải thích ngắn")
        if pred == 1:
            st.write("Mẫu này được xếp vào nhóm nguy cơ vì xác suất dự đoán lớn hơn hoặc bằng ngưỡng đã chọn.")
        else:
            st.write("Mẫu này chưa bị xếp vào nhóm nguy cơ vì xác suất dự đoán thấp hơn ngưỡng đã chọn.")

        st.markdown("### Một số giá trị đầu vào sau khi kiểm soát biên")
        show_cols = [c for c in available_input_cols if c in input_df.columns]
        st.dataframe(input_df[show_cols], use_container_width=True)


# =========================================================
# TRANG 3 - ĐÁNH GIÁ HIỆU NĂNG
# =========================================================
elif page == "Đánh giá hiệu năng":
    st.markdown("<div class='section-title'>Trang 3: Đánh giá hiệu năng</div>", unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)

    with m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Accuracy</div>
                <div class="metric-value">{eval_result['accuracy']:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Precision (Nguy cơ)</div>
                <div class="metric-value">{eval_result['precision_pos']:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Recall (Nguy cơ)</div>
                <div class="metric-value">{eval_result['recall_pos']:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">F1-score (Nguy cơ)</div>
                <div class="metric-value">{eval_result['f1_pos']:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">AUC-ROC</div>
                <div class="metric-value">{eval_result['auc']:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="card">
            <h3>0. Giải thích các chỉ số</h3>
            <ul>
                <li><b>Accuracy:</b> Tỷ lệ dự đoán đúng trên toàn bộ tập kiểm tra.</li>
                <li><b>Precision:</b> Trong các bệnh nhân bị dự đoán là nguy cơ, có bao nhiêu người đúng.</li>
                <li><b>Recall:</b> Trong các bệnh nhân thực sự nguy cơ, mô hình phát hiện được bao nhiêu người.</li>
                <li><b>F1-score:</b> Chỉ số cân bằng giữa Precision và Recall.</li>
                <li><b>AUC-ROC:</b> Khả năng phân biệt hai lớp dựa trên xác suất dự đoán.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("<div class='card'><h3>1. Confusion Matrix</h3></div>", unsafe_allow_html=True)
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        cm = confusion_matrix(y_test, y_pred)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Không nguy cơ", "Nguy cơ"],
        )
        disp.plot(ax=ax_cm, cmap="Blues", colorbar=False, include_values=False)

        labels = [["TN", "FP"],
                ["FN", "TP"]]

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax_cm.text(
                    j, i,
                    f"{labels[i][j]}\n{cm[i, j]}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    color="white" if cm[i, j] > cm.max()/2 else "#163b73"
                )

        ax_cm.set_title("Ma trận nhầm lẫn", fontsize=14, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_cm)

    with col_right:
        st.markdown("<div class='card'><h3>2. ROC Curve</h3></div>", unsafe_allow_html=True)
        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax_roc.plot(fpr, tpr, label=f"AUC = {eval_result['auc']:.4f}")
        ax_roc.plot([0, 1], [0, 1], linestyle="--")
        ax_roc.set_title("Đường cong ROC", fontsize=14, fontweight="bold")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.legend()
        plt.tight_layout()
        st.pyplot(fig_roc)

    st.markdown("<div class='card'><h3>3. Báo cáo phân loại</h3></div>", unsafe_allow_html=True)
    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=["Không nguy cơ", "Nguy cơ"],
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df, use_container_width=True)

    tn, fp, fn, tp = cm.ravel()
    st.markdown(
        f"""
        <div class="card">
            <h3>4. Phân tích sai số tổng quát</h3>
            <ul>
                <li><b>TN = {tn}</b>: dự đoán đúng bệnh nhân không nguy cơ.</li>
                <li><b>FP = {fp}</b>: dự đoán nhầm bệnh nhân không nguy cơ thành nguy cơ.</li>
                <li><b>FN = {fn}</b>: bỏ sót bệnh nhân nguy cơ thực sự.</li>
                <li><b>TP = {tp}</b>: phát hiện đúng bệnh nhân nguy cơ.</li>
            </ul>
            <p>Trong bài toán y tế, chỉ số <b>Recall</b> và số <b>FN</b> rất quan trọng vì bỏ sót bệnh nhân nguy cơ có thể ảnh hưởng đến việc theo dõi và can thiệp sớm.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='card'><h3>5. Chi tiết một trường hợp dự đoán sai</h3></div>", unsafe_allow_html=True)

    mis_idx = np.where(y_test.to_numpy() != y_pred)[0]

    if len(mis_idx) > 0:
        first_mis = mis_idx[0]

        wrong_sample = X_test.iloc[first_mis].copy()
        true_val = y_test.iloc[first_mis]
        true_label = "Nguy cơ" if true_val == 1 else "Không nguy cơ"
        pred_label_text = "Nguy cơ" if y_pred[first_mis] == 1 else "Không nguy cơ"
        pred_prob = y_prob[first_mis]

        summary_df = pd.DataFrame({
            "Nhãn thật": [true_label],
            "Nhãn dự đoán": [pred_label_text],
            "Xác suất dự đoán nguy cơ": [f"{pred_prob:.2%}"],
        })
        st.dataframe(summary_df, use_container_width=True)

        st.markdown(
            """
            <div class="card">
                <h4>Dữ liệu gốc của mẫu bị dự đoán sai</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )

        detail_df = pd.DataFrame(wrong_sample).T
        st.dataframe(detail_df, use_container_width=True)

        st.markdown(
            f"""
            <div class="card">
                <h4>Nhận định</h4>
                <p>Trường hợp này mô hình dự đoán sai vì nhãn thật là <b>{true_label}</b> nhưng mô hình lại dự đoán thành <b>{pred_label_text}</b>.
                Điều này cho thấy một số hồ sơ bệnh nhân có đặc trưng gần nhau giữa hai lớp, nên mô hình Naive Bayes vẫn có thể nhầm lẫn khi phân loại.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.success("Không có mẫu dự đoán sai trên tập kiểm tra ở ngưỡng hiện tại.")


# =========================================================
# TRANG 4 - SO SÁNH & CẢI TIẾN
# =========================================================
elif page == "So sánh & cải tiến":
    st.markdown("<div class='section-title'>Trang 4: So sánh & cải tiến</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="card">
            <h3>1. Mô hình đang sử dụng</h3>
            <p><b>Multinomial Naive Bayes</b> là một biến thể của Naive Bayes phù hợp hơn khi đặc trưng đầu vào ở dạng không âm sau khi mã hoá và chuẩn hoá. Kết hợp thêm bước <b>Calibration</b> giúp xác suất đầu ra bớt cực đoan hơn.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    compare_df = pd.DataFrame({
        "Mô hình": ["Gaussian Naive Bayes", "Multinomial Naive Bayes", "Bernoulli Naive Bayes"],
        "Ưu điểm": [
            "Đơn giản, nhanh với dữ liệu số liên tục",
            "Phù hợp với đặc trưng không âm sau tiền xử lý",
            "Phù hợp với dữ liệu nhị phân 0/1",
        ],
        "Hạn chế": [
            "Không hợp khi dữ liệu có nhiều đặc trưng one-hot và phân phối không chuẩn",
            "Vẫn có giả định độc lập giữa các đặc trưng",
            "Kém phù hợp nếu có nhiều đặc trưng số liên tục",
        ],
        "Mức phù hợp với bài này": ["Thấp hơn", "Phù hợp hơn", "Khá phù hợp"],
    })

    st.markdown("<div class='card'><h3>2. Bảng so sánh tổng quan trong họ Naive Bayes</h3></div>", unsafe_allow_html=True)
    st.dataframe(compare_df, use_container_width=True)

    st.markdown(
        """
        <div class="card">
            <h3>3. Hạn chế của bài hiện tại</h3>
            <ul>
                <li>Mô hình Naive Bayes giả định các đặc trưng độc lập, trong khi dữ liệu y tế thường có mối liên hệ giữa các biến.</li>
                <li>Xác suất dự đoán vẫn phụ thuộc khá nhiều vào chất lượng dữ liệu đầu vào.</li>
                <li>Một số trường hợp cực trị vẫn có thể làm xác suất tăng hoặc giảm mạnh.</li>
                <li>Chưa thực hiện chọn lọc đặc trưng chuyên sâu hơn.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="card">
            <h3>4. Hướng cải tiến</h3>
            <ul>
                <li>Tối ưu thêm bộ đặc trưng đầu vào.</li>
                <li>Thử đánh giá thêm bằng cross-validation.</li>
                <li>Tối ưu threshold theo mục tiêu tăng Recall cho lớp nguy cơ.</li>
                <li>Phân tích thêm các trường hợp dự đoán sai để tinh chỉnh dữ liệu.</li>
                <li>Cân nhắc BernoulliNB nếu muốn nhấn mạnh nhóm đặc trưng nhị phân.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="card">
            <h3>5. Kết luận</h3>
            <p>Bài toán này là bài toán <b>phân loại nhị phân</b> với mục tiêu phát hiện bệnh nhân có nguy cơ tái nhập viện sớm. Thay vì dùng Gaussian Naive Bayes dễ cho xác suất cực đoan, phiên bản hiện tại sử dụng <b>Multinomial Naive Bayes kết hợp Calibration</b> để kết quả dự đoán hợp lý và ổn định hơn, nhưng vẫn giữ đúng định hướng Naive Bayes của đề tài.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
