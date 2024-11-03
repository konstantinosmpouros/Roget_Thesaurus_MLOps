import pathlib
import os
import prediction_model

PACKAGE_ROOT = pathlib.Path(prediction_model.__file__).resolve().parent

# Dataset Variables
DATAPATH = os.path.join(PACKAGE_ROOT,"datasets")
FILE_NAME = "Roget's_Words.csv"
CLASS_TEST_FILE = "Class_test.csv"
SECTION_TEST_FILE = "Section_test.csv"
TARGET_CLASS = 'Class'
TARGET_SECTION = 'Section'


# Model Variables
CLASS_MODEL_NAME = 'xgboost_class_pipeline.joblib'
CLASS_MODEL_PATH = os.path.join(PACKAGE_ROOT,'trained_models/class')

SECTION_MODEL_NAME = 'xgboost_section_pipeline.joblib'
SECTION_MODEL_PATH = os.path.join(PACKAGE_ROOT,'trained_models/section')

# Faiss database
FAISS_NAME = 'Train_embeddings.faiss'

