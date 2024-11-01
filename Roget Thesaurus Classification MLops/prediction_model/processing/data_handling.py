import os 
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))


from prediction_model.config import config

# Load the dataset
def load_dataset():
    filepath = os.path.join(config.DATAPATH, config.FILE_NAME)
    _data = pd.read_csv(filepath)
    return _data

# Separate X and y according to prediction target
def separate_data_section(data, target):
    try:
        if target == 'class':
            X = data.drop(config.TARGET_CLASS, axis=1)
            y= data[config.TARGET_CLASS]
        elif target == 'section':
            X = data.drop(config.TARGET_SECTION, axis=1)
            y= data[config.TARGET_SECTION]
        else:
            raise Exception('You must define a the target (class or section)')
        
        return X, y
    except Exception as ex:
        print(ex)

# Split into training and testing sets
def split_data(X, y, test_size=0.2, random_state=33):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

#Serialization
def save_pipeline(pipeline_to_save, target):
    try:
        if target == 'class':
            save_path = os.path.join(config.CLASS_MODEL_PATH, config.MODEL_NAME)
        elif target == 'section':
            save_path = os.path.join(config.SECTION_MODEL_PATH, config.MODEL_NAME)
        else:
            raise Exception('You must define a the target (class or section)')

        print(save_path)
        joblib.dump(pipeline_to_save, save_path)
        print(f"Model has been saved under the name {config.MODEL_NAME}")
    except Exception as ex:
        print(ex)

#Deserialization
def load_pipeline(target):
    try:
        if target == 'class':
            save_path = os.path.join(config.CLASS_MODEL_PATH, config.MODEL_NAME)
        elif target == 'section':
            save_path = os.path.join(config.SECTION_MODEL_PATH, config.MODEL_NAME)
        else:
            raise Exception('You must define a the target (class or section)')
        
        model_loaded = joblib.load(save_path)
        print(f"Model has been loaded")
        return model_loaded
    except Exception as ex:
        print(ex)
