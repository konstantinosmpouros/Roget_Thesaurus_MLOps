import os
import sys
from pathlib import Path

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.calibration import LabelEncoder
import faiss

# Load the dataset
def load_dataset():
    filepath = os.path.join(PACKAGE_ROOT, "datasets/Roget's_Words.csv")
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

# Encode y labels
def encode_y_data(y):
    encoder = LabelEncoder()
    return pd.DataFrame(encoder.fit_transform(y.values.ravel()))

def load_embeddings(target):
    if target == 'train':
        path = os.path.join(PACKAGE_ROOT, 'datasets/Train_word_embeddings.faiss')
    elif target == 'test':
        path = os.path.join(PACKAGE_ROOT, 'datasets/Test_word_embeddings.faiss')
    
    index = faiss.read_index(path)
    d = index.d
    embeddings = np.zeros((index.ntotal, d), dtype=np.float32)

    for i in range(index.ntotal):
        embeddings[i] = index.reconstruct(i)

    return pd.DataFrame(embeddings)
