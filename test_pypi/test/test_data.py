from pathlib import Path
import os
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config
from prediction_model.processing.data_handling import load_dataset, separate_data, split_data, encode_y_data

datasets = {
    config.CLASS_TEST_FILE : [config.TARGET_CLASS],
    config.SECTION_TEST_FILE : [config.TARGET_SECTION],
    config.FILE_NAME : [config.TARGET_CLASS, config.TARGET_SECTION]
}

# Test data loading
def test_data_loading():
    for dataset_name in datasets.keys():
        dataset = load_dataset(dataset_name)
        assert dataset is not None, f"Dataset {dataset_name} should not be None"
        assert len(dataset) > 0, f"Dataset {dataset_name} should contain data"
        assert len(dataset.columns) == 3 or len(dataset.columns) == 7, f"Dataset {dataset_name} should contain data"

# Test data separation into X and y
def test_data_separation():
    for name in datasets.keys():
        dataset = load_dataset(name)
        for target in datasets[name]:
            X, y = separate_data(dataset, target)

            assert X is not None, f"Features (X) on {name} should not be None"
            assert y is not None, f"Labels (y) on {name} should not be None"
            assert len(X) == len(y), f"Features and labels on {name} should have the same number of samples"
            assert len(y.columns) == 1, f"Labels (y) should contain only the label data"
            assert len(X.columns) == 1, f"Data (X) should contain only the Final_Words column"

# Test data spliting into train and test
def test_split_data():
    for name in datasets.keys():
        dataset = load_dataset(name)
        for target in datasets[name]:
            X, y = separate_data(dataset, target)
            X_train, X_test, y_train, y_test = split_data(X, y)

            assert X_train is not None and X_test is not None, "Split data should not be None"
            assert y_train is not None and y_test is not None, "Split labels should not be None"
            assert len(X_train) + len(X_test) == len(X), "Total samples should match after split on features"
            assert len(X_train) > len(X_test), "Training set should be larger than test set"
            assert len(y_train) + len(y_test) == len(y), "Total samples should match after split on labels"
            assert len(y_train) > len(y_test), "Training set should be larger than test set"

# Test y data encoding
def test_y_encoding():
    for name in datasets.keys():
        dataset = load_dataset(name)
        for target in datasets[name]:
            X, y = separate_data(dataset, target)

            y_encoded = encode_y_data(y)
            unique_encoded_values = set(y_encoded)
            unique_original_values = set(y.iloc[:, 0])

            assert y_encoded is not None, f"Encoded labels should not be None for {name}"
            assert len(y_encoded) == len(y), f"Encoded labels should have the same length as original labels for {name}"
            assert len(unique_encoded_values) == len(unique_original_values), (
                f"Number of unique encoded values should match the number of unique original labels for {name}"
            )
