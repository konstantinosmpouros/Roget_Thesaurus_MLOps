import os
from pathlib import Path
import sys
import joblib
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

import prediction_model.processing.preprocessing as pp 
from prediction_model.config import config


class CustomPipeline():
    def __init__(self, target):
        self.target = target
        self.pipeline = None

        if self.target == config.TARGET_CLASS:
            self.save_path = os.path.join(config.CLASS_MODEL_PATH, config.CLASS_MODEL_NAME)
        else:
            self.save_path = os.path.join(config.SECTION_MODEL_PATH, config.SECTION_MODEL_NAME)

    def create_pipeline(self):
        self.pipeline = Pipeline([
                    ('Gemma2B_Embeddings', pp.Gemma2B_Embeddings()),
                    ('StandarScaling', pp.StandarScaling()),
                    ('DimensionalityReduction', pp.DimensionalityReduction()),
                    ('XGBoost', XGBClassifier(n_jobs=-1, random_state=33))
        ])

    def save_pipeline(self):
        joblib.dump(self.pipeline, self.save_path)
        print('Model has been saved successfully!!')
        print('Path:', self.save_path)

    def load_pipeline(self):
        self.pipeline = joblib.load(self.save_path)
        print(f"Model has been loaded")

