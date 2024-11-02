from pathlib import Path
import os
import sys

from prediction_model.processing.data_handling import load_dataset, separate_data, split_data
from prediction_model.config import config
from pipeline import RogetPipeline


PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))


def train_pipeline():
    df = load_dataset()
    X, y = separate_data(df, 'class')

if __name__=='__main__':
    train_pipeline()