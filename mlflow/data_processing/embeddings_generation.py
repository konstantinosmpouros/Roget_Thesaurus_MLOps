import argparse
from pathlib import Path
import os
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from data_processing.preprocessing import CustomPipeline
from data_processing.data_handling import load_dataset, separate_data, split_data

def generate():
    data = load_dataset()
    X, y = separate_data(data, 'Class')
    X_train, X_test, y_train, _ = split_data(X, y)
    
    print('Generateing the training embeddings!')
    pipeline = CustomPipeline()
    pipeline.pipeline.fit(X_train, y_train)

    print('Generateing the test embeddings!')
    pipeline.pipeline['SaveEmbeddings'].change_to_test()
    pipeline.pipeline.transform(X_test)
