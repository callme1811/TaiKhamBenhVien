import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)

st.set_page_config(
    page_title="Dự đoán tái nhập viện bệnh nhân tiểu đường",
    page_icon="🏥",
    layout="wide"
)

st.title("Ứng dụng mô hình Naive Bayes dự đoán nguy cơ tái nhập viện")

# =========================
# CACHE
# =========================
@st.cache_data
def load_data():
    file_path = r"D:\Machine with python\customer\TaiKhamBenhVien\TaiKhamBenhVien\hospital_readmissions.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return df
    return None

@st.cache_resource
def train_model(df):
    data = df.copy()

    target_col = "readmitted"
    X = data.drop(columns=[target_col])
    y = data[target_col].astype(str)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

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
        X, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    model = GaussianNB()
    model.fit(X_train_processed, y_train)

    y_pred = model.predict(X_test_processed)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_processed)[:, 1]
    else:
        y_prob = None

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_prob) if y_prob is not None else None
    }

    return {
        "model": model,
        "preprocessor": preprocessor,
        "label_encoder": le,
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "metrics": metrics,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols
    }

# =========================
# LOAD DATA
# =========================
df = load_data()

if df is None:
    st.warning("Không tìm thấy file hospital_readmissions.csv trong thư mục dự án.")
    uploaded_file = st.file_uploader("Tải lên file hospital_readmissions.csv", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.stop()

artifacts = train_model(df)

# =========================
# SIDEBAR
# =========================
page = st.sidebar.radio(
    "Chọn trang",
    ["Trang 1: Giới thiệu & EDA", "Trang 2: Triển khai mô hình", "Trang 3: Đánh giá hiệu năng"]
)

# =========================
# PAGE 1
# =========================
if page == "Trang 1: Giới thiệu & EDA":
    st.header("Trang 1: Giới thiệu & Khám phá dữ liệu (EDA)")

    st.subheader("1. Thông tin bài toán")
    st.markdown("""
**Tên đề tài:** Ứng dụng mô hình Naive Bayes để phân loại nguy cơ tái nhập viện của bệnh nhân đái tháo đường

**Họ tên SV:** ....................................................  
**MSSV:** ....................................................  

**Mô tả ngắn gọn giá trị thực tiễn:**  
Mô hình hỗ trợ dự đoán sớm bệnh nhân có nguy cơ tái nhập viện, từ đó giúp bác sĩ và bệnh viện theo dõi sát hơn, can thiệp kịp thời và tối ưu phân bổ nguồn lực.
""")

    st.subheader("2. Xem dữ liệu")
    st.write("Kích thước dữ liệu:", df.shape)
    st.dataframe(df.head(10), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.write("Số dòng:", df.shape[0])
    with c2:
        st.write("Số cột:", df.shape[1])

    st.subheader("3. Kiểm tra dữ liệu thiếu")
    missing_df = pd.DataFrame({
        "Cột": df.columns,
        "Số giá trị thiếu": df.isnull().sum().values
    })
    st.dataframe(missing_df, use_container_width=True)

    st.subheader("4. Các biểu đồ phân tích")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Biểu đồ 1: Phân bố nhãn readmitted**")
        fig1, ax1 = plt.subplots()
        df["readmitted"].value_counts().plot(kind="bar", ax=ax1)
        ax1.set_xlabel("readmitted")
        ax1.set_ylabel("Số lượng")
        ax1.set_title("Phân bố nhãn")
        st.pyplot(fig1)

    with col2:
        st.markdown("**Biểu đồ 2: Phân bố theo tuổi**")
        fig2, ax2 = plt.subplots()
        df["age"].value_counts().sort_index().plot(kind="bar", ax=ax2)
        ax2.set_xlabel("Nhóm tuổi")
        ax2.set_ylabel("Số lượng")
        ax2.set_title("Phân bố nhóm tuổi")
        st.pyplot(fig2)

    st.markdown("**Biểu đồ 3: Số ngày nằm viện theo nhãn readmitted**")
    fig3, ax3 = plt.subplots()
    avg_time = df.groupby("readmitted")["time_in_hospital"].mean()
    avg_time.plot(kind="bar", ax=ax3)
    ax3.set_xlabel("readmitted")
    ax3.set_ylabel("Thời gian nằm viện trung bình")
    ax3.set_title("So sánh time_in_hospital theo nhãn")
    st.pyplot(fig3)

    st.subheader("5. Nhận xét dữ liệu")
    st.markdown("""
- Dữ liệu có cột mục tiêu là **readmitted** với 2 lớp: **yes / no**.
- Bộ dữ liệu gồm cả biến số và biến phân loại như: **age, time_in_hospital, n_lab_procedures, medical_specialty, glucose_test, A1Ctest...**
- Dữ liệu này phù hợp cho bài toán **phân loại nhị phân**.
- Các đặc trưng như số lần nhập viện, số lần cấp cứu, thời gian nằm viện và xét nghiệm có khả năng ảnh hưởng tới nguy cơ tái nhập viện.
""")

# =========================
# PAGE 2
# =========================
elif page == "Trang 2: Triển khai mô hình":
    st.header("Trang 2: Triển khai mô hình")

    st.subheader("1. Nhập dữ liệu bệnh nhân")

    input_data = {}

    for col in artifacts["numeric_cols"]:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())

        input_data[col] = st.number_input(
            f"{col}",
            min_value=min_val,
            max_value=max_val,
            value=mean_val
        )

    for col in artifacts["categorical_cols"]:
        options = sorted(df[col].astype(str).unique().tolist())
        default_index = 0
        input_data[col] = st.selectbox(f"{col}", options, index=default_index)

    input_df = pd.DataFrame([input_data])

    st.subheader("2. Dự đoán")
    if st.button("Dự đoán nguy cơ tái nhập viện"):
        X_input = artifacts["preprocessor"].transform(input_df)
        pred = artifacts["model"].predict(X_input)[0]

        if hasattr(artifacts["model"], "predict_proba"):
            prob = artifacts["model"].predict_proba(X_input)[0][1]
        else:
            prob = None

        label = artifacts["label_encoder"].inverse_transform([pred])[0]

        if label == "yes":
            st.error(f"Kết quả dự đoán: Bệnh nhân CÓ nguy cơ tái nhập viện")
        else:
            st.success(f"Kết quả dự đoán: Bệnh nhân KHÔNG có nguy cơ tái nhập viện")

        if prob is not None:
            st.write(f"Xác suất tái nhập viện: **{prob:.2%}**")
            st.progress(float(prob))

    st.subheader("3. Mô tả xử lý logic")
    st.markdown("""
- Dữ liệu đầu vào được tách thành biến số và biến phân loại.
- Biến phân loại được mã hóa bằng **OneHotEncoder**.
- Mô hình sử dụng **Gaussian Naive Bayes** để phân loại nguy cơ tái nhập viện.
- Quy trình huấn luyện dùng chia tập **Train/Test = 80/20**.
""")

# =========================
# PAGE 3
# =========================
elif page == "Trang 3: Đánh giá hiệu năng":
    st.header("Trang 3: Đánh giá hiệu năng (Evaluation)")

    metrics = artifacts["metrics"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    c2.metric("Precision", f"{metrics['precision']:.4f}")
    c3.metric("Recall", f"{metrics['recall']:.4f}")
    c4.metric("F1-score", f"{metrics['f1']:.4f}")
    c5.metric("AUC-ROC", f"{metrics['auc']:.4f}" if metrics["auc"] is not None else "N/A")

    st.subheader("1. Confusion Matrix")
    fig_cm, ax_cm = plt.subplots()
    cm = confusion_matrix(artifacts["y_test"], artifacts["y_pred"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax_cm)
    st.pyplot(fig_cm)

    if artifacts["y_prob"] is not None:
        st.subheader("2. ROC Curve")
        fpr, tpr, _ = roc_curve(artifacts["y_test"], artifacts["y_prob"])
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f"AUC = {metrics['auc']:.4f}")
        ax_roc.plot([0, 1], [0, 1], linestyle="--")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("ROC Curve")
        ax_roc.legend()
        st.pyplot(fig_roc)

    st.subheader("3. Nhận xét")
    st.markdown("""
- **Accuracy** cho biết tỷ lệ dự đoán đúng trên toàn bộ dữ liệu kiểm tra.
- **Precision** phản ánh mức độ chính xác khi mô hình dự đoán bệnh nhân có nguy cơ tái nhập viện.
- **Recall** cho biết mô hình phát hiện được bao nhiêu bệnh nhân thật sự có nguy cơ.
- **F1-score** cân bằng giữa Precision và Recall.
- **AUC-ROC** giúp đánh giá khả năng phân biệt 2 lớp của mô hình.

**Kết luận:**  
Mô hình Naive Bayes có thể dùng để hỗ trợ phân loại nguy cơ tái nhập viện. Tuy nhiên, để tăng hiệu quả thực tế, có thể so sánh thêm với Logistic Regression, Random Forest hoặc XGBoost.
""")