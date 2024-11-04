import os 
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys
from sklearn.calibration import LabelEncoder

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config

# Load the dataset
def load_dataset(file_name):
    filepath = os.path.join(config.DATAPATH, file_name)
    data = pd.read_csv(filepath, encoding='latin')
    return data

# Separate X and y according to prediction target
def separate_data(data, target):
    X = data[['Final_Words']]
    y = data[[target]]
    return X, y

# Split into training and testing sets
def split_data(X, y, test_size=0.2, random_state=33):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def encode_y_data(y):
    encoder = LabelEncoder()
    return encoder.fit_transform(y.values.ravel())

