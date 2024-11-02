import pathlib
import os
import prediction_model

PACKAGE_ROOT = pathlib.Path(prediction_model.__file__).resolve().parent

# Dataset Variables
DATAPATH = os.path.join(PACKAGE_ROOT,"datasets")
FILE_NAME = "Roget's_Words.csv"
TARGET_CLASS = 'Class'
TARGET_SECTION = 'Section'

# Model Variables
MODEL_NAME = 'xgboost.joblib'
CLASS_MODEL_PATH = os.path.join(PACKAGE_ROOT,'trained_models/class')
SECTION_MODEL_PATH = os.path.join(PACKAGE_ROOT,'trained_models/section')

