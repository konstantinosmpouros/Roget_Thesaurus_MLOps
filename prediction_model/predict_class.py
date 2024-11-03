from pathlib import Path
import os
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config
from prediction_model.processing.data_handling import load_dataset, separate_data, encode_y_data
from prediction_model.pipeline import CustomPipeline

def score_class():
    test_data = load_dataset(config.CLASS_TEST_FILE)
    X, y = separate_data(test_data, config.TARGET_CLASS)
    y = encode_y_data(y)

    class_pipeline = CustomPipeline(config.TARGET_CLASS)
    class_pipeline.load_pipeline()
    class_pipeline.pipeline.score(X, y)


if __name__=='__main__':
    score_class()
