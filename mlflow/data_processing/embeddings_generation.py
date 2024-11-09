import argparse
from pathlib import Path
import os
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from data_processing.preprocessing import CustomPipeline
from data_processing.data_handling import load_dataset, separate_data

if __name__ == '__main__':
    data = load_dataset()
    X, y = separate_data(data, 'Class')
    
    pipeline = CustomPipeline()
    pipeline.pipeline.fit(X, y)
