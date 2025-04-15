from pathlib import Path
import os
import sys
from sklearn.metrics import accuracy_score

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config
from prediction_model.processing.data_handling import load_dataset, separate_data
from prediction_model.pipeline import CustomPipeline

def score_section():
    # Load test data
    test_data = load_dataset(config.SECTION_TEST_FILE)
    X, y = separate_data(test_data, config.TARGET_SECTION)

    # Load trained pipeline
    section_pipeline = CustomPipeline(config.TARGET_SECTION).load_pipeline()

    # Scoring
    y_pred = section_pipeline.pipeline.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print('Class Pipeline score:', accuracy)


if __name__=='__main__':
    score_section()
