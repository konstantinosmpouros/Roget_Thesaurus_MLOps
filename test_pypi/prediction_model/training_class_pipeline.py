from pathlib import Path
import os
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.processing.data_handling import load_dataset, separate_data, split_data
from prediction_model.config import config
from prediction_model.pipeline import CustomPipeline


def train_class_pipeline():
    # Load Data
    df = load_dataset(config.FILE_NAME)
    X, y = separate_data(df, config.TARGET_CLASS)

    # Split to train and test
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Save the test set for future scoring
    test_data = X_test.copy()
    test_data[config.TARGET_CLASS] = y_test
    test_data.to_csv(os.path.join(config.DATAPATH, config.CLASS_TEST_FILE))

    # Training
    class_pipeline = CustomPipeline(config.TARGET_CLASS)
    class_pipeline.create_pipeline()
    class_pipeline.pipeline.fit(X_train, y_train)
    
    # Save pipeline
    class_pipeline.save_pipeline()

    # Scoring
    print('Class Pipeline score:', class_pipeline.pipeline.score(X_test, y_test))

if __name__=='__main__':
    train_class_pipeline()