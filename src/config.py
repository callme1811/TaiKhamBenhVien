import os

PAGE_TITLE = "Phân loại nguy cơ tái nhập viện sớm"
PAGE_ICON = "🏥"
LAYOUT = "wide"
INITIAL_SIDEBAR_STATE = "expanded"

DATA_PATH = "data/hospital_readmissions.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "multinomial_nb_calibrated_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor_nb.pkl")

THRESHOLD = 0.5
RANDOM_STATE = 42
TEST_SIZE = 0.2
CALIBRATION_CV = 3
NB_ALPHA = 1.0

NAV_OPTIONS = [
    "Giới thiệu & EDA",
    "Triển khai mô hình",
    "Đánh giá hiệu năng",
    "So sánh & cải tiến",
]

SELECTED_INPUT_COLS = [
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
