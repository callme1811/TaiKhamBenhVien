#  Phân loại nguy cơ tái nhập viện bệnh nhân đái tháo đường

##  Giới thiệu
Ứng dụng sử dụng Machine Learning (Naive Bayes) để dự đoán nguy cơ tái nhập viện sớm của bệnh nhân dựa trên dữ liệu bệnh án.

Ứng dụng được xây dựng bằng Streamlit nhằm:
- Khám phá dữ liệu (EDA)
- Dự đoán nguy cơ tái nhập viện
- Đánh giá hiệu năng mô hình

---

##  Công nghệ sử dụng
- Python
- Scikit-learn
- Streamlit
- Pandas, NumPy
- Matplotlib

---

##  Cấu trúc thư mục
```
TaiKhamBenhVien/
├── app.py
├── hospital_readmissions.csv
├── models/
├── requirements.txt
├── README.md
```

---

##  Cách chạy
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

##  Deploy Streamlit Cloud
- Chọn repo GitHub
- Main file: app.py
- Nhấn Deploy

---

##  Mô hình
- Multinomial Naive Bayes + Calibration
- Nhị phân:
  - 1: Tái nhập viện
  - 0: Không

---
##  Metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
