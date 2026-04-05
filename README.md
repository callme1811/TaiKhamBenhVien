# Phân loại nguy cơ tái nhập viện sớm bằng Naive Bayes

Đây là bản project được giữ theo đúng logic của `app.py` gốc.

## Lưu ý quan trọng
- `app.py` là file chính và được giữ đúng theo code gốc.
- Thư mục `src/` chỉ được tạo thêm để đúng form cấu trúc project như yêu cầu.
- Khi chạy thực tế, ưu tiên dùng trực tiếp `app.py`.

## Cấu trúc thư mục
```
readmission_nb_project_exact/
├── app.py
├── README.md
├── requirements.txt
├── data/
│   └── hospital_readmissions.csv
├── models/
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

## Cài thư viện
```bash
pip install -r requirements.txt
```

## Chạy ứng dụng
```bash
streamlit run app.py
```

## Dữ liệu
Đặt file dữ liệu tại:
```
data/hospital_readmissions.csv
```

## Mô hình trong app.py
- Thuật toán: `Multinomial Naive Bayes + CalibratedClassifierCV`
- Bài toán: phân loại nhị phân
- Nhãn: `1 = tái nhập viện sớm`, `0 = không tái nhập viện sớm`
- Ngưỡng dự đoán cố định: `0.50`
