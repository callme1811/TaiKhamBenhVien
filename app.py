import os
import joblib
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
    roc_curve,
    classification_report
)

st.set_page_config(
    page_title="Phân loại nguy cơ tái nhập viện",
    page_icon="🏥",
    layout="wide"
)

st.title("Phân loại nguy cơ tái nhập viện bệnh nhân đái tháo đường bằng Naive Bayes")

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "naive_bayes_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")


@st.cache_data
def load_data():
    return pd.read_csv("hospital_readmissions.csv")


@st.cache_resource
def prepare_and_train_model(df):
    data = df.copy()

    target_col = "readmitted"
    X = data.drop(columns=[target_col])
    y = data[target_col].astype(str)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

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
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    model = GaussianNB()
    model.fit(X_train_processed, y_train)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)

    return model, preprocessor, label_encoder, X_train, X_test, y_train, y_test, categorical_cols, numeric_cols


@st.cache_resource
def load_or_train_model(df):
    if (
        os.path.exists(MODEL_PATH)
        and os.path.exists(PREPROCESSOR_PATH)
        and os.path.exists(LABEL_ENCODER_PATH)
    ):
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)

        data = df.copy()
        target_col = "readmitted"
        X = data.drop(columns=[target_col])
        y = data[target_col].astype(str)
        y_encoded = label_encoder.transform(y)

        categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
        numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_encoded,
            test_size=0.2,
            random_state=42,
            stratify=y_encoded
        )

        return model, preprocessor, label_encoder, X_train, X_test, y_train, y_test, categorical_cols, numeric_cols

    return prepare_and_train_model(df)


df = load_data()

model, preprocessor, label_encoder, X_train, X_test, y_train, y_test, categorical_cols, numeric_cols = load_or_train_model(df)

X_test_processed = preprocessor.transform(X_test)
y_pred = model.predict(X_test_processed)

y_prob = None
auc = None
if len(label_encoder.classes_) == 2:
    y_prob = model.predict_proba(X_test_processed)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
    "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
    "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    "auc": auc
}

page = st.sidebar.radio(
    "Chọn trang",
    [
        "Trang 1: Giới thiệu & EDA",
        "Trang 2: Triển khai mô hình",
        "Trang 3: Đánh giá hiệu năng"
    ]
)

if page == "Trang 1: Giới thiệu & EDA":
    st.header("Trang 1: Giới thiệu & Khám phá dữ liệu (EDA)")

    st.subheader("1. Thông tin bài toán")
    st.markdown("""
**Tên đề tài:** Phân loại nguy cơ tái nhập viện của bệnh nhân đái tháo đường bằng Naive Bayes  

**Họ tên SV:** Nguyễn Trọng Quý  
**MSSV:** 22T1020719

**Mô tả ngắn gọn giá trị thực tiễn:**  
Mô hình hỗ trợ phân loại sớm bệnh nhân có nguy cơ tái nhập viện, từ đó giúp bác sĩ và bệnh viện theo dõi sát hơn, can thiệp kịp thời và tối ưu phân bổ nguồn lực.
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
        "Tên cột": df.columns,
        "Số giá trị thiếu": df.isnull().sum().values
    })
    st.dataframe(missing_df, use_container_width=True)

    st.subheader("4. Các biểu đồ phân tích")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Biểu đồ 1: Phân bố nhãn readmitted**")
        fig1, ax1 = plt.subplots()
        df["readmitted"].value_counts().plot(kind="bar", ax=ax1)
        ax1.set_xlabel("readmitted")
        ax1.set_ylabel("Số lượng")
        ax1.set_title("Phân bố nhãn")
        st.pyplot(fig1)

    with c2:
        st.markdown("**Biểu đồ 2: Phân bố theo tuổi**")
        fig2, ax2 = plt.subplots()
        df["age"].value_counts().sort_index().plot(kind="bar", ax=ax2)
        ax2.set_xlabel("Nhóm tuổi")
        ax2.set_ylabel("Số lượng")
        ax2.set_title("Phân bố nhóm tuổi")
        st.pyplot(fig2)

    st.markdown("**Biểu đồ 3: Thời gian nằm viện trung bình theo nhãn readmitted**")
    fig3, ax3 = plt.subplots()
    avg_time = df.groupby("readmitted")["time_in_hospital"].mean()
    avg_time.plot(kind="bar", ax=ax3)
    ax3.set_xlabel("readmitted")
    ax3.set_ylabel("Thời gian nằm viện trung bình")
    ax3.set_title("So sánh time_in_hospital theo nhãn")
    st.pyplot(fig3)

    st.subheader("5. Nhận xét dữ liệu")
    st.markdown("""
- Dữ liệu có cột mục tiêu là **readmitted**, dùng để xác định bệnh nhân có nguy cơ tái nhập viện hay không.
- Bộ dữ liệu gồm cả biến số và biến phân loại như: **age, time_in_hospital, n_lab_procedures, medical_specialty, glucose_test, A1Ctest...**
- Đây là bộ dữ liệu phù hợp cho bài toán **phân loại** trong lĩnh vực y tế.
- Một số đặc trưng như thời gian nằm viện, số lần nhập viện, số lần cấp cứu và kết quả xét nghiệm có thể ảnh hưởng đến nguy cơ tái nhập viện.
- Dữ liệu có ý nghĩa thực tiễn vì hỗ trợ bệnh viện sàng lọc sớm các trường hợp nguy cơ cao để có kế hoạch theo dõi và điều trị phù hợp.
""")

elif page == "Trang 2: Triển khai mô hình":
    st.header("Trang 2: Triển khai mô hình")

    st.subheader("1. Nhập dữ liệu bệnh nhân")
    input_data = {}

    for col in numeric_cols:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())

        input_data[col] = st.number_input(
            label=col,
            min_value=min_val,
            max_value=max_val,
            value=mean_val
        )

    for col in categorical_cols:
        options = sorted(df[col].astype(str).unique().tolist())
        input_data[col] = st.selectbox(col, options)

    input_df = pd.DataFrame([input_data])

    st.subheader("2. Xử lý logic")
    st.markdown("""
- Ứng dụng sử dụng mô hình **Naive Bayes đã được huấn luyện trước** và lưu dưới dạng file `.pkl`.
- Bộ tiền xử lý dữ liệu cũng được lưu riêng để bảo đảm dữ liệu nhập vào được xử lý giống lúc huấn luyện.
- Dữ liệu đầu vào được tách thành biến số và biến phân loại.
- Biến số được điền giá trị thiếu bằng **median**.
- Biến phân loại được điền giá trị thiếu bằng **most frequent** và mã hóa bằng **OneHotEncoder**.
""")

    st.subheader("3. Hiển thị kết quả")
    if st.button("Dự đoán nguy cơ tái nhập viện"):
        X_input = preprocessor.transform(input_df)
        pred = model.predict(X_input)[0]
        label = label_encoder.inverse_transform([pred])[0]

        if str(label).lower() in ["yes", "readmitted", "1", "<30", ">30"]:
            st.error("Kết quả phân loại: Bệnh nhân có nguy cơ tái nhập viện")
        else:
            st.success("Kết quả phân loại: Bệnh nhân không có nguy cơ tái nhập viện")

        if len(label_encoder.classes_) == 2:
            prob = model.predict_proba(X_input)[0][1]
            st.write(f"Xác suất thuộc nhóm nguy cơ tái nhập viện: **{prob:.2%}**")
            st.progress(float(prob))

elif page == "Trang 3: Đánh giá hiệu năng":
    st.header("Trang 3: Đánh giá hiệu năng")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    c2.metric("Precision", f"{metrics['precision']:.4f}")
    c3.metric("Recall", f"{metrics['recall']:.4f}")
    c4.metric("F1-score", f"{metrics['f1']:.4f}")
    c5.metric("AUC-ROC", f"{metrics['auc']:.4f}" if metrics["auc"] is not None else "N/A")

    st.subheader("1. Confusion Matrix")
    fig_cm, ax_cm = plt.subplots()
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax_cm)
    st.pyplot(fig_cm)

    if y_prob is not None:
        st.subheader("2. ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f"AUC = {metrics['auc']:.4f}")
        ax_roc.plot([0, 1], [0, 1], linestyle="--")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("ROC Curve")
        ax_roc.legend()
        st.pyplot(fig_roc)

    st.subheader("3. Báo cáo phân loại")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df, use_container_width=True)

    st.subheader("4. Phân tích sai số")
    st.markdown("""
- Mô hình có thể dự đoán sai ở những trường hợp bệnh nhân có đặc điểm gần giống nhau giữa hai nhóm tái nhập viện và không tái nhập viện.
- Nguyên nhân là Naive Bayes giả định các đặc trưng độc lập với nhau, trong khi dữ liệu y tế thực tế thường có mối liên hệ giữa nhiều yếu tố như số lần nhập viện, thời gian nằm viện và kết quả xét nghiệm.
- Nếu dữ liệu bị mất cân bằng giữa các lớp, mô hình cũng có thể thiên về lớp xuất hiện nhiều hơn.
- Để cải thiện, có thể thử:
  - Cân bằng dữ liệu bằng oversampling hoặc undersampling
  - Chọn lọc đặc trưng quan trọng hơn
  - So sánh với các mô hình khác như Logistic Regression, Random Forest hoặc XGBoost
""")

    st.subheader("5. Nhận xét")
    st.markdown("""
- **Accuracy** cho biết tỷ lệ dự đoán đúng trên toàn bộ tập kiểm tra.
- **Precision** phản ánh mức độ chính xác khi mô hình dự đoán bệnh nhân thuộc nhóm có nguy cơ tái nhập viện.
- **Recall** cho biết mô hình phát hiện được bao nhiêu trường hợp thật sự có nguy cơ.
- **F1-score** là chỉ số cân bằng giữa Precision và Recall.
- **AUC-ROC** thể hiện khả năng phân biệt giữa các lớp của mô hình.

**Kết luận:**  
Mô hình Naive Bayes có ưu điểm là đơn giản, tốc độ huấn luyện nhanh và phù hợp với bài toán phân loại dữ liệu y tế. Tuy nhiên, do giả định các đặc trưng độc lập với nhau nên độ chính xác có thể chưa tối ưu so với một số mô hình mạnh hơn.
""")