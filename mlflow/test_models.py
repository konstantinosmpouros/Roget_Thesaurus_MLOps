import mlflow
import pandas as pd
import numpy as np
import faiss

import os
import sys
from pathlib import Path

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(str(PACKAGE_ROOT))

def load_test_embeddings_sample():
    test_path = os.path.join(PACKAGE_ROOT, 'datasets/Test_word_embeddings.faiss')
    index = faiss.read_index(test_path)
    d = index.d
    embeddings = np.zeros((index.ntotal, d), dtype=np.float32)

    for i in range(index.ntotal):
        embeddings[i] = index.reconstruct(i)

    embeddings_sample = pd.DataFrame(embeddings).iloc[:2, :]
    return embeddings_sample

def load_y_labels(target):
    path = os.path.join(PACKAGE_ROOT, f'datasets/{target}_TEST.csv')
    return pd.read_csv(path)

# Load test embeddings
embeddings_sample = load_test_embeddings_sample()

# Load true labels
class_labels = load_y_labels('CLASS')
section_labels = load_y_labels('SECTION')

# Models uri
class_model_uri = 'runs:/a5ea513ef1b348f6ae209c85d8c0e7db/LGBMClassifier' # The uris will be different if you run the project
section_model_uri = 'runs:/50700ad801d54235a2fbc5aa459ae0fe/LGBMClassifier' # The uris will be different if you run the project

# Load models as a PyFuncModel
class_model = mlflow.pyfunc.load_model(class_model_uri)
section_model = mlflow.pyfunc.load_model(section_model_uri)

# Make the predictions
print('The Class predictions are: ', class_model.predict(embeddings_sample))
print('The Class true labels are: ', class_labels.iloc[:2, 2].to_list())
print()
print('The Class predictions are: ', section_model.predict(embeddings_sample))
print('The Class true labels are: ', section_labels.iloc[:2, 2].to_list())