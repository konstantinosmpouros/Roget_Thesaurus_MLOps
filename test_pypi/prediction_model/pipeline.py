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
                    ('StandarScaling_1', pp.StandardScaling()),
                    ('DimensionalityReduction', pp.DimensionalityReduction()),
                    ('StandarScaling_2', pp.StandardScaling()),
                    ('LGBMClassifier', pp.LGBM()),
        ])

    # Save the pipeline object to a file using joblib
    def save_pipeline(self):
        if not os.path.exists(self.save_path):
            joblib.dump(self.pipeline, self.save_path)
            print('Model has been saved successfully!!')
        else:
            joblib.dump(self.pipeline, self.save_path)
            print('Existing model has been replace with the new one successfully!!')

    # Load the pipeline from the file
    def load_pipeline(self):
        if os.path.exists(self.save_path):
            self.pipeline = joblib.load(self.save_path)
            print(f"Model has been loaded")
            return self
        else:
            print(f'No saved pipeline found. Running {self.target} training.')
            if self.target == 'Class':
                training_class_pipeline = __import__('prediction_model.training_class_pipeline')
            else:
                training_section_pipeline = __import__('prediction_model.training_section_pipeline')

            self.pipeline = joblib.load(self.save_path)
            print(f"Model has been loaded")
            return self
