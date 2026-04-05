# Phân loại nguy cơ tái nhập viện sớm từ dữ liệu bệnh án bằng Naive Bayes nhằm hỗ trợ phát hiện bệnh nhân rủi ro cao.
Mục tiêu của ứng dụng là dự đoán **nguy cơ tái nhập viện sớm** của bệnh nhân đái tháo đường bằng **Multinomial Naive Bayes kết hợp Calibration**.

## Cấu trúc thư mục

```bash
readmission_nb_project/
├── app.py
├── README.md
├── data/
│   └── hospital_readmissions.csv
├── models/
│   ├── multinomial_nb_calibrated_model.pkl
│   └── preprocessor_nb.pkl
└── src/
    ├── __init__.py
    ├── config.py
    ├── dataset.py
    ├── inference.py
    ├── predict.py
    ├── preprocessing.py
    ├── train.py
    ├── trainer_utils.py
    └── utils.py
```

## Ý nghĩa từng file trong `src`

- `config.py`: chứa cấu hình chung như đường dẫn dữ liệu, model, threshold, menu điều hướng.
- `dataset.py`: đọc dữ liệu, sinh giá trị mặc định cho form nhập liệu, tạo giới hạn input.
- `preprocessing.py`: làm sạch dữ liệu, chuẩn hóa target, tạo feature engineering, build preprocessor.
- `train.py`: huấn luyện hoặc load lại mô hình đã lưu.
- `trainer_utils.py`: đánh giá mô hình, tính các metric, confusion matrix, ROC, classification report.
- `inference.py`: nhận dữ liệu đầu vào bệnh nhân và trả về kết quả dự đoán.
- `predict.py`: file trung gian để export hàm dự đoán.
- `utils.py`: CSS giao diện, hero banner, hàm phụ trợ.

## Thuật toán đang dùng

- **Mô hình chính**: `MultinomialNB(alpha=1.0)`
- **Hiệu chỉnh xác suất**: `CalibratedClassifierCV(method="sigmoid", cv=3)`
- **Tiền xử lý**:
  - Numeric: `SimpleImputer(strategy="median")` + `MinMaxScaler()`
  - Categorical: `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(handle_unknown="ignore")`
- **Ngưỡng dự đoán cố định**: `0.50`

## Luồng xử lý chính

```text
Dữ liệu CSV
→ clean_raw_data()
→ add_feature_engineering()
→ build_preprocessor()
→ train/test split
→ Multinomial Naive Bayes
→ Calibration
→ predict_proba()
→ threshold = 0.50
→ nhãn 0 / 1
```

## Cách chạy

### 1. Cài thư viện

```bash
pip install streamlit pandas numpy matplotlib scikit-learn joblib
```

### 2. Đặt dữ liệu

Chép file dữ liệu vào đúng đường dẫn:

```bash
data/hospital_readmissions.csv
```

### 3. Chạy ứng dụng

```bash
streamlit run app.py
```
