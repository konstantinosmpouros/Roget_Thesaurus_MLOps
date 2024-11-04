from sklearn.base import BaseEstimator,TransformerMixin

from pathlib import Path
import os
import sys
import pandas as pd

from sklearn.preprocessing import StandardScaler
from umap import umap_ as umap

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))


class Gemma7B_Embeddings(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.model_name = "google/gemma-1.1-7b-it"

    def load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map='auto',
            torch_dtype=torch.bfloat16, 
        )
        model.eval()
        self.move_to_gpu(model, tokenizer)
        return tokenizer, model

    def move_to_gpu(self, model, tokenizer):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            tokenizer.to(device)
            model.to(device)
        except Exception as ex:
            pass
    
    def set_seed(self, seed=33):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def fit(self, X, y=None):
        return self

    def transform(self, X, batch_size=100):
        embeddings = []
        tokenizer, model = self.load_model()
        self.set_seed()

        for start in tqdm(range(0, len(X), batch_size)):
            batch = X.iloc[start:start + batch_size, 0].tolist() 

            batch_tokenized  = tokenizer(batch,
                                 truncation=True,
                                 padding='max_length',
                                 max_length=20,
                                 return_tensors='pt')

            with torch.no_grad():
                outputs = model(**batch_tokenized, output_hidden_states=True)

            last_hidden_states = outputs.hidden_states[-1]
            batch_word_embedding  = last_hidden_states.mean(dim=1)
            embeddings.extend(batch_word_embedding.cpu().float().numpy())
        
        print('Embeddings have been created successfully!!')
        return pd.DataFrame(embeddings)

class StandarScaling(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X)
        return self

    def transform(self, X):
        scaled_x = self.scaler.transform(X)
        print('Embeddings has been Standar Scaled successfully!!')
        return pd.DataFrame(scaled_x)

class DimensionalityReduction(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.dim_reduction = umap.UMAP(n_components=300, n_jobs=-1)

    def fit(self, X, y=None):
        self.dim_reduction.fit(X)
        return self

    def transform(self, X):
        reduced_embeddings =  self.dim_reduction.transform(X)
        print('Embeddings has been reduced to 300 dimensions successfully!!')
        return pd.DataFrame(reduced_embeddings)

