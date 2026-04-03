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
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    classification_report
)
from sklearn.naive_bayes import GaussianNB

st.set_page_config(
    page_title="Phân loại nguy cơ tái nhập viện",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: "Segoe UI", sans-serif;
}

.block-container {
    max-width: 1320px;
    padding-top: 1.2rem;
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
    padding: 30px 32px;
    box-shadow: 0 12px 28px rgba(0,0,0,0.25);
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 20px;
}

.hero-title {
    font-size: 44px;
    font-weight: 800;
    line-height: 1.15;
    color: white;
    margin-bottom: 8px;
}

.hero-subtitle {
    font-size: 16px;
    color: #dbeafe;
    line-height: 1.6;
}

.section-title {
    font-size: 32px;
    font-weight: 800;
    margin: 8px 0 16px 0;
}

.card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 18px 20px;
    margin-bottom: 16px;
    box-shadow: 0 8px 18px rgba(0,0,0,0.12);
}

.metric-card {
    background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 18px 14px;
    text-align: center;
    box-shadow: 0 8px 18px rgba(0,0,0,0.12);
}

.metric-label {
    font-size: 15px;
    color: #cbd5e1;
    margin-bottom: 8px;
}

.metric-value {
    font-size: 28px;
    font-weight: 800;
    color: white;
}

.small-muted {
    color: #94a3b8;
    font-size: 14px;
}

.info-chip {
    display: inline-block;
    background: rgba(37,99,235,0.18);
    color: #bfdbfe;
    border: 1px solid rgba(59,130,246,0.35);
    padding: 6px 12px;
    border-radius: 999px;
    font-size: 13px;
    margin-right: 8px;
    margin-bottom: 8px;
}

hr {
    border: none;
    height: 1px;
    background: rgba(255,255,255,0.08);
    margin: 14px 0;
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
    box-shadow: 0 10px 20px rgba(37,99,235,0.28);
}

div.stButton > button:hover {
    background: linear-gradient(135deg, #1d4ed8, #1e40af);
    color: white;
}

div[data-testid="stMetric"] {
    background: transparent;
    border: none;
    box-shadow: none;
}

[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-box">
    <div class="hero-title">Phân loại nguy cơ tái nhập viện của bệnh nhân đái tháo đường bằng Naive Bayes</div>
    <div class="hero-subtitle">
        Ứng dụng hỗ trợ khám phá dữ liệu, dự đoán nguy cơ tái nhập viện và đánh giá hiệu năng mô hình
        trên hồ sơ bệnh nhân bằng Streamlit.
    </div>
</div>
""", unsafe_allow_html=True)

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "naive_bayes_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")


@st.cache_data
def load_data():
    df = pd.read_csv("hospital_readmissions.csv")
    df["readmitted"] = df["readmitted"].replace({
        "<30": "YES",
        ">30": "YES",
        "NO": "NO"
    })
    return df


def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    if all(col in data.columns for col in ["n_outpatient", "n_inpatient", "n_emergency"]):
        data["total_visits"] = (
            data["n_outpatient"].fillna(0)
            + data["n_inpatient"].fillna(0)
            + data["n_emergency"].fillna(0)
        )

    if all(col in data.columns for col in ["n_inpatient", "n_outpatient"]):
        data["visit_ratio"] = data["n_inpatient"] / (data["n_outpatient"] + 1)

    if all(col in data.columns for col in ["time_in_hospital", "n_lab_procedures"]):
        data["severity"] = data["time_in_hospital"] * data["n_lab_procedures"]

    if all(col in data.columns for col in ["n_medications", "n_procedures"]):
        data["med_ratio"] = data["n_medications"] / (data["n_procedures"] + 1)

    if all(col in data.columns for col in ["n_medications", "time_in_hospital"]):
        data["care_intensity"] = data["n_medications"] / (data["time_in_hospital"] + 1)

    if "age" in data.columns:
        data["is_elderly"] = data["age"].astype(str).str.contains("70|80|90").astype(int)

    if "time_in_hospital" in data.columns:
        data["long_stay"] = (data["time_in_hospital"] > 7).astype(int)

    if "n_medications" in data.columns:
        data["high_med"] = (data["n_medications"] > 10).astype(int)

    if all(col in data.columns for col in ["n_lab_procedures", "n_procedures"]):
        data["total_procedures"] = data["n_lab_procedures"] + data["n_procedures"]

    return data


@st.cache_resource(show_spinner=False)
def prepare_and_train_model(df):
    data = add_feature_engineering(df)

    target_col = "readmitted"
    X = data.drop(columns=[target_col])
    y = data[target_col].astype(str)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object", "string"]).columns.tolist()

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    if hasattr(X_train_processed, "toarray"):
        X_train_processed = X_train_processed.toarray()
    if hasattr(X_test_processed, "toarray"):
        X_test_processed = X_test_processed.toarray()

    model = GaussianNB()
    model.fit(X_train_processed, y_train)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)

    return (
        model,
        preprocessor,
        label_encoder,
        X_train,
        X_test,
        y_train,
        y_test,
        categorical_cols,
        numeric_cols
    )


@st.cache_resource(show_spinner=False)
def load_or_train_model(df):
    data = add_feature_engineering(df)
    target_col = "readmitted"
    X = data.drop(columns=[target_col])
    y = data[target_col].astype(str)

    categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object", "string"]).columns.tolist()

    if (
        os.path.exists(MODEL_PATH)
        and os.path.exists(PREPROCESSOR_PATH)
        and os.path.exists(LABEL_ENCODER_PATH)
    ):
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)

        y_encoded = label_encoder.transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_encoded,
            test_size=0.2,
            random_state=42,
            stratify=y_encoded
        )

        return (
            model,
            preprocessor,
            label_encoder,
            X_train,
            X_test,
            y_train,
            y_test,
            categorical_cols,
            numeric_cols
        )

    return prepare_and_train_model(df)


df = load_data()

(
    model,
    preprocessor,
    label_encoder,
    X_train,
    X_test,
    y_train,
    y_test,
    categorical_cols,
    numeric_cols
) = load_or_train_model(df)

X_test_processed = preprocessor.transform(X_test)
if hasattr(X_test_processed, "toarray"):
    X_test_processed = X_test_processed.toarray()

y_prob = model.predict_proba(X_test_processed)[:, 1]
y_pred = (y_prob > 0.5).astype(int)
auc = roc_auc_score(y_test, y_prob)

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
    "recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
    "f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
    "auc": auc
}

st.sidebar.markdown("## Điều hướng")
st.sidebar.markdown("<div class='small-muted'>Chọn nội dung muốn xem</div>", unsafe_allow_html=True)

page = st.sidebar.radio(
    "",
    [
        "Giới thiệu & EDA",
        "Triển khai mô hình",
        "Đánh giá hiệu năng",
        "So sánh & cải tiến"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Thông tin mô hình")
st.sidebar.markdown("**Thuật toán:** Gaussian Naive Bayes")
st.sidebar.markdown("**Bài toán:** Phân loại nhị phân")
st.sidebar.markdown("**Nguồn dữ liệu:** hospital_readmissions.csv")

if page == "Giới thiệu & EDA":
    st.markdown("<div class='section-title'>Trang 1: Giới thiệu & Khám phá dữ liệu (EDA)</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>1. Thông tin bài toán</h3>
        <p><b>Tên đề tài:</b> Phân loại nguy cơ tái nhập viện của bệnh nhân đái tháo đường bằng Naive Bayes nhằm hỗ trợ dự đoán sớm và tối ưu theo dõi điều trị</p>
        <p><b>Họ tên SV:</b> Nguyễn Trọng Quý<br><b>MSSV:</b> 22T1020719</p>
        <p><b>Mô tả ngắn gọn giá trị thực tiễn:</b><br>
        Mô hình hỗ trợ phân loại sớm bệnh nhân có nguy cơ tái nhập viện, từ đó giúp bác sĩ và bệnh viện theo dõi sát hơn, can thiệp kịp thời và tối ưu phân bổ nguồn lực.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='card'><h3>2. Xem dữ liệu</h3></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"<div class='info-chip'>Số dòng: {df.shape[0]}</div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='info-chip'>Số cột: {df.shape[1]}</div>", unsafe_allow_html=True)
    st.dataframe(df.head(10), width="stretch")

    st.markdown("<div class='card'><h3>3. Kiểm tra dữ liệu thiếu</h3></div>", unsafe_allow_html=True)
    missing_df = pd.DataFrame({
        "Tên cột": df.columns,
        "Số giá trị thiếu": df.isnull().sum().values
    })
    st.dataframe(missing_df, width="stretch")

    st.markdown("<div class='card'><h3>4. Các biểu đồ phân tích</h3></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots()
        df["readmitted"].value_counts().plot(kind="bar", ax=ax1)
        ax1.set_title("Phân bố nhãn readmitted")
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        df["age"].value_counts().sort_index().plot(kind="bar", ax=ax2)
        ax2.set_title("Phân bố nhóm tuổi")
        st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    df.groupby("readmitted")["time_in_hospital"].mean().plot(kind="bar", ax=ax3)
    ax3.set_title("Thời gian nằm viện trung bình theo nhãn")
    st.pyplot(fig3)

    st.markdown("""
    <div class="card">
        <h3>5. Nhận xét dữ liệu</h3>
        <ul>
            <li>Dữ liệu có cột mục tiêu là <b>readmitted</b>.</li>
            <li>Bộ dữ liệu gồm cả biến số và biến phân loại.</li>
            <li>Đây là bài toán phân loại nhị phân sau khi gộp nhãn.</li>
            <li>Một số đặc trưng như thời gian nằm viện, số lần nhập viện, số xét nghiệm và số thuốc có ảnh hưởng đến kết quả.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>6. Giải thích dữ liệu</h3>
        <ul>
            <li>Nhãn <b>readmitted</b> đã được gộp thành <b>YES</b> và <b>NO</b>.</li>
            <li>Có thêm các đặc trưng mới bằng feature engineering để tăng thông tin cho mô hình.</li>
            <li>Naive Bayes phù hợp làm mô hình cơ bản vì đơn giản, dễ triển khai và tốc độ huấn luyện nhanh.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

elif page == "Triển khai mô hình":
    st.markdown("<div class='section-title'>Trang 2: Triển khai mô hình</div>", unsafe_allow_html=True)

    left, right = st.columns([1.2, 0.8])

    with left:
        st.markdown("<div class='card'><h3>1. Nhập dữ liệu bệnh nhân</h3></div>", unsafe_allow_html=True)
        note = st.text_input("Ghi chú bệnh nhân (tuỳ chọn)")
        input_data = {}

        for col in numeric_cols:
            min_val = float(X_train[col].min())
            max_val = float(X_train[col].max())
            mean_val = float(X_train[col].mean())

            input_data[col] = st.number_input(
                label=col,
                min_value=min_val,
                max_value=max_val,
                value=mean_val
            )

        for col in categorical_cols:
            options = sorted(X_train[col].astype(str).unique().tolist())
            input_data[col] = st.selectbox(col, options)

    with right:
        st.markdown("""
        <div class="card">
            <h3>2. Xử lý logic</h3>
            <p>- Dữ liệu được xử lý bằng <b>SimpleImputer</b> và <b>OneHotEncoder</b>.</p>
            <p>- Có áp dụng <b>feature engineering</b> để tạo thêm các đặc trưng hỗ trợ dự đoán.</p>
            <p>- Mô hình sử dụng <b>Gaussian Naive Bayes</b>.</p>
            <p>- Sau khi mã hóa, dữ liệu được chuyển sang dạng số để phù hợp với GaussianNB.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <h3>3. Pipeline</h3>
            <p>Dữ liệu → Gộp label → Feature Engineering → Tiền xử lý → Train/Test → Naive Bayes → Dự đoán</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <h3>4. Cấu hình mô hình</h3>
            <p><b>Thuật toán:</b> GaussianNB</p>
            <p><b>Ngưỡng dự đoán:</b> 0.5</p>
        </div>
        """, unsafe_allow_html=True)

    input_df = pd.DataFrame([input_data]) if len(input_data) > 0 else None

    st.markdown("<div class='card'><h3>5. Hiển thị kết quả</h3></div>", unsafe_allow_html=True)

    if st.button("Dự đoán nguy cơ tái nhập viện"):
        input_df = add_feature_engineering(input_df)
        X_input = preprocessor.transform(input_df)
        if hasattr(X_input, "toarray"):
            X_input = X_input.toarray()

        prob = model.predict_proba(X_input)[0][1]
        pred = 1 if prob > 0.5 else 0
        label = label_encoder.inverse_transform([pred])[0]

        if note:
            st.info(f"Ghi chú bệnh nhân: {note}")

        r1, r2 = st.columns([1, 1])

        with r1:
            if str(label).lower() == "yes":
                st.error("Kết quả: Bệnh nhân có nguy cơ tái nhập viện")
            else:
                st.success("Kết quả: Bệnh nhân không có nguy cơ tái nhập viện")

        with r2:
            st.metric("Xác suất nguy cơ", f"{prob:.2%}")

        st.progress(float(prob))

elif page == "Đánh giá hiệu năng":
    st.markdown("<div class='section-title'>Trang 3: Đánh giá hiệu năng</div>", unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Accuracy</div>
            <div class="metric-value">{metrics['accuracy']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Precision</div>
            <div class="metric-value">{metrics['precision']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Recall</div>
            <div class="metric-value">{metrics['recall']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">F1-score</div>
            <div class="metric-value">{metrics['f1']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">AUC-ROC</div>
            <div class="metric-value">{metrics['auc']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>0. Giải thích chỉ số</h3>
        <ul>
            <li><b>Accuracy:</b> tỷ lệ dự đoán đúng.</li>
            <li><b>Precision:</b> độ chính xác khi dự đoán lớp nguy cơ.</li>
            <li><b>Recall:</b> khả năng phát hiện đúng bệnh nhân nguy cơ.</li>
            <li><b>F1-score:</b> cân bằng giữa Precision và Recall.</li>
            <li><b>AUC-ROC:</b> khả năng phân biệt giữa hai lớp.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("<div class='card'><h3>1. Confusion Matrix</h3></div>", unsafe_allow_html=True)
        fig_cm, ax_cm = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
        disp.plot(ax=ax_cm)
        st.pyplot(fig_cm)

    with col_right:
        st.markdown("<div class='card'><h3>2. ROC Curve</h3></div>", unsafe_allow_html=True)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f"AUC = {metrics['auc']:.4f}")
        ax_roc.plot([0, 1], [0, 1], linestyle="--")
        ax_roc.legend()
        st.pyplot(fig_roc)

    st.markdown("<div class='card'><h3>3. Báo cáo phân loại</h3></div>", unsafe_allow_html=True)
    report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True, zero_division=0)).transpose()
    st.dataframe(report_df, width="stretch")

    st.markdown("""
    <div class="card">
        <h3>4. Phân tích sai số</h3>
        <ul>
            <li>Naive Bayes có ưu điểm là đơn giản và tốc độ nhanh.</li>
            <li>Tuy nhiên mô hình giả định các đặc trưng độc lập với nhau, nên với dữ liệu y tế thực tế độ chính xác có thể không quá cao.</li>
            <li>Việc gộp label và feature engineering giúp mô hình ổn định hơn so với dùng dữ liệu gốc.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>5. Nhận xét</h3>
        <p>Naive Bayes là lựa chọn phù hợp làm mô hình cơ sở cho bài toán này. Ứng dụng đáp ứng được yêu cầu phân loại nguy cơ tái nhập viện và dễ triển khai.</p>
    </div>
    """, unsafe_allow_html=True)

elif page == "So sánh & cải tiến":
    st.markdown("<div class='section-title'>Trang 4: So sánh & cải tiến</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>1. So sánh các mô hình</h3>
        <ul>
            <li><b>Naive Bayes:</b> đơn giản, nhanh nhưng độ chính xác thường thấp hơn.</li>
            <li><b>Random Forest:</b> khá tốt với dữ liệu bảng.</li>
            <li><b>XGBoost:</b> mạnh với boosting và ổn định.</li>
            <li><b>LightGBM:</b> nhanh, mạnh và phù hợp với dữ liệu bảng lớn.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='card'><h3>2. Bảng so sánh tổng quan</h3></div>", unsafe_allow_html=True)
    compare_df = pd.DataFrame({
        "Mô hình": ["Naive Bayes", "Random Forest", "XGBoost", "LightGBM"],
        "Độ chính xác": ["Trung bình", "Khá", "Cao", "Cao"],
        "Tốc độ": ["Rất nhanh", "Trung bình", "Trung bình", "Nhanh"],
        "Độ phức tạp": ["Thấp", "Trung bình", "Cao", "Cao"]
    })
    st.dataframe(compare_df, width="stretch")

    st.markdown("""
    <div class="card">
        <h3>3. Hạn chế</h3>
        <ul>
            <li>Dữ liệu y tế có nhiều nhiễu.</li>
            <li>Feature hiện có chưa phản ánh đầy đủ mức độ bệnh.</li>
            <li>Naive Bayes giả định độc lập đặc trưng, trong khi dữ liệu thực tế thường có mối liên hệ giữa các biến.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>4. Hướng cải tiến</h3>
        <ul>
            <li>Thử các mô hình mạnh hơn như Random Forest, XGBoost hoặc LightGBM.</li>
            <li>Bổ sung thêm dữ liệu thật từ nguồn khác.</li>
            <li>Tạo thêm đặc trưng mạnh hơn.</li>
            <li>Tối ưu ngưỡng dự đoán và cách chọn đặc trưng.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>5. Giá trị thực tiễn mở rộng</h3>
        <ul>
            <li>Hỗ trợ dự đoán sớm bệnh nhân có nguy cơ tái nhập viện.</li>
            <li>Giúp bệnh viện phân bổ nguồn lực tốt hơn.</li>
            <li>Có thể tích hợp vào hệ thống quản lý bệnh viện thực tế.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)