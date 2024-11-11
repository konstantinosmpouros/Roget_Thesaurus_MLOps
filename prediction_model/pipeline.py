import os
from pathlib import Path
import sys
import joblib
from sklearn.pipeline import Pipeline


PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

import prediction_model.processing.preprocessing as pp 
from prediction_model.config import config


class CustomPipeline():
    def __init__(self, target):
        # Set the target (either Class or Section)
        self.target = target
        self.pipeline = None

        # Define the path where the trained model will be saved or loaded from, based on the target
        if self.target == config.TARGET_CLASS:
            self.save_path = os.path.join(config.CLASS_MODEL_PATH, config.CLASS_MODEL_NAME)
        else:
            self.save_path = os.path.join(config.SECTION_MODEL_PATH, config.SECTION_MODEL_NAME)

    def create_pipeline(self):
        # Define the pipeline (A sequence of transformations and the classifier)
        self.pipeline = Pipeline([
                    ('Gemma_2B_Embeddings', pp.Gemma_2B_Embeddings()),
                    ('StandarScaling', pp.StandardScaling()),
                    ('DimensionalityReduction', pp.DimensionalityReduction()),
                    ('LGBMClassifier', pp.LGBM())
        ])

    # Save the pipeline object to a file using joblib
    def save_pipeline(self):
        joblib.dump(self.pipeline, self.save_path)
        print('Model has been saved successfully!!')
        print('Path:', self.save_path)

    # Load the pipeline from the file
    def load_pipeline(self):
        self.pipeline = joblib.load(self.save_path)
        print(f"Model has been loaded")
        return self

