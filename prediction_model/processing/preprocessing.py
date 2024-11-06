from sklearn.base import BaseEstimator,TransformerMixin

from pathlib import Path
import os
import sys
import pandas as pd

from sklearn.preprocessing import StandardScaler
from umap import UMAP

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))


class Gemma2B_Embeddings(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.model_name = "google/gemma-1.1-2b-it"

    def load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map='auto',
            torch_dtype=torch.bfloat16, 
        )

        model.eval()
        return tokenizer, model

    def fit(self, X, y=None):
        return self

    def transform(self, X, batch_size=100):
        embeddings = []
        tokenizer, model = self.load_model()

        for start in tqdm(range(0, len(X), batch_size)):
            batch = X.iloc[start:start + batch_size, 0].tolist() 

            batch_tokenized  = tokenizer(batch,
                                         truncation=True,
                                         padding='max_length',
                                         max_length=20,
                                         return_tensors='pt').to('cuda')

            with torch.no_grad():
                outputs = model(**batch_tokenized, output_hidden_states=True)

            last_hidden_states = outputs.hidden_states[-1]
            batch_word_embedding  = last_hidden_states.mean(dim=1)
            embeddings.extend(batch_word_embedding.cpu().float().numpy())
        
        print('Embeddings have been created successfully!!')
        return pd.DataFrame(embeddings)

class StandardScaling(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X)
        return self

    def transform(self, X):
        scaled_x = self.scaler.transform(X)
        print('Embeddings have been Standar Scaled successfully!!')
        return pd.DataFrame(scaled_x)

class DimensionalityReduction(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.dim_reduction = UMAP(n_components=300, n_jobs=-1)

    def fit(self, X, y=None):
        self.dim_reduction.fit(X)
        return self

    def transform(self, X):
        reduced_embeddings =  self.dim_reduction.transform(X)
        print('Embeddings have been reduced to 300 dimensions successfully!!')
        return pd.DataFrame(reduced_embeddings)

