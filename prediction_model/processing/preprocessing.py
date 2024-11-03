from sklearn.base import BaseEstimator,TransformerMixin

from pathlib import Path
import os
import sys
import pandas as pd
import numpy as np

from umap import umap_ as umap

import faiss

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config


class Gemma2B_Embeddings(BaseEstimator, TransformerMixin):
    def __init__(self):
        model_name = "google/gemma-1.1-2b-it"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map='auto',
            torch_dtype=torch.bfloat16, 
        )
        self.model.eval()
        self.move_to_gpu()

    def move_to_gpu(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            self.model.to(device)
        except Exception as ex:
            pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, batch_size=100):
        embeddings = []
        for start in tqdm(range(0, len(X), batch_size)):
            batch = X.iloc[start:start + batch_size, 0].tolist() 

            batch_tokenized  = self.tokenizer(batch,
                                 truncation=True,
                                 padding='max_length',
                                 max_length=20,
                                 return_tensors='pt').to('cuda')

            with torch.no_grad():
                outputs = self.model(**batch_tokenized, output_hidden_states=True)

            last_hidden_states = outputs.hidden_states[-1]
            batch_word_embedding  = last_hidden_states.mean(dim=1)
            embeddings.extend(batch_word_embedding.cpu().float().numpy())

        return pd.DataFrame(embeddings)

class SaveEmbedding(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.save_path = os.path.join(config.DATAPATH, config.FAISS_NAME)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not os.path.exists(self.save_path):
            d = X.shape[1]
            index = faiss.IndexFlatL2(d)
            index.add(np.ascontiguousarray(X.values.astype('float32')))
            faiss.write_index(index, self.save_path)
            
            print('Embeddings have been saved successfully!!')
            print('Path: ', self.save_path)
        else:
            print('Embeddings already exist!!')
        return X

class DimensionalityReduction(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.dim_reduction = umap.UMAP(n_components=300, random_state=33)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        reduced_embeddings =  self.dim_reduction.fit_transform(X)
        print('Embeddings has been reduced to 300 dimensions successfully!!')
        return pd.DataFrame(reduced_embeddings)

