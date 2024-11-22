import os
import sys
from pathlib import Path

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from data_processing.preprocessing import CustomPipeline

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.calibration import LabelEncoder
import faiss

# Load the dataset
def load_dataset(filename="Roget's_Words"):
    filepath = os.path.join(PACKAGE_ROOT, f"datasets/{filename}.csv")
    data = pd.read_csv(filepath, encoding='latin')

    # Cleaning
    data['Section'] = data['Section'].str.replace(r'^SECTION\s+\w+\.\s*', '', regex=True).str.strip()
    data['Class'] = data['Class'].str.replace(r'^CLASS\s+\w+\s+', '', regex=True).str.strip()
    return data

# Separate X and y according to prediction target
def separate_data(data, target):
    X = data[['Final_Words']].astype(str)
    y = data[[target]]
    return X, y

# Split into training and testing sets
def split_data(X, y, test_size=0.2, random_state=33):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Encode y labels
def encode_y_data(y):
    encoder = LabelEncoder()
    return pd.DataFrame(encoder.fit_transform(y.values.ravel()))

def generate_embeddings():
    data = load_dataset()
    X, y = separate_data(data, 'Class')
    X_train, X_test, y_train, _ = split_data(X, y)
    
    print('Generateing the training embeddings!')
    pipeline = CustomPipeline()
    pipeline.pipeline.fit(X_train, y_train)

    print('Generateing the test embeddings!')
    pipeline.pipeline['SaveEmbeddings'].change_to_test()
    pipeline.pipeline.transform(X_test)

def load_embeddings():
    train_path = os.path.join(PACKAGE_ROOT, 'datasets/Train_word_embeddings.faiss')
    test_path = os.path.join(PACKAGE_ROOT, 'datasets/Test_word_embeddings.faiss')
    
    if os.path.exists(train_path) and os.path.exists(test_path):
        # Load train embeddings
        index = faiss.read_index(train_path)
        d = index.d
        train_embeddings = np.zeros((index.ntotal, d), dtype=np.float32)

        for i in range(index.ntotal):
            train_embeddings[i] = index.reconstruct(i)
        print(f'The training embeddigns has been successfully loaded!!')

        # Load test embeddings
        index = faiss.read_index(test_path)
        d = index.d
        test_embeddings = np.zeros((index.ntotal, d), dtype=np.float32)

        for i in range(index.ntotal):
            test_embeddings[i] = index.reconstruct(i)
        print(f'The test embeddigns has been successfully loaded!!')

        return pd.DataFrame(train_embeddings), pd.DataFrame(test_embeddings)

    else:
        print(f'The embeddigns doesnt exist!!')
        generate_embeddings()
        return load_embeddings()
