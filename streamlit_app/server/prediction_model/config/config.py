import pathlib
import os
import prediction_model

PACKAGE_ROOT = pathlib.Path(prediction_model.__file__).resolve().parent

# Dataset Variables
TARGET_CLASS = 'Class'
TARGET_SECTION = 'Section'


# Model Variables
CLASS_MODEL_NAME = 'class_pipeline.joblib'
CLASS_MODEL_PATH = os.path.join(PACKAGE_ROOT,'trained_models/class')

SECTION_MODEL_NAME = 'section_pipeline.joblib'
SECTION_MODEL_PATH = os.path.join(PACKAGE_ROOT,'trained_models/section')


