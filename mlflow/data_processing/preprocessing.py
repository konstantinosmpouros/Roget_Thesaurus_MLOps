from pathlib import Path
import os
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from sklearn.preprocessing import StandardScaler
from umap import UMAP
import faiss
from sklearn.pipeline import Pipeline


class Gemma_2B_Embeddings(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.model_name = "google/gemma-1.1-2b-it"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map='auto',
            torch_dtype=torch.bfloat16, 
        )
        
        self.model.eval()

    def set_seed(self):
        torch.manual_seed(33)
        torch.cuda.manual_seed_all(33)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def fit(self, X, y=None):
        return self

    def transform(self, X, batch_size=100):
        self.set_seed()
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
        self.n_components = 300
        self.dim_reduction = UMAP(n_components=self.n_components, random_state=33)

    def fit(self, X, y=None):
        self.dim_reduction.fit(X)
        return self

    def transform(self, X):
        reduced_embeddings =  self.dim_reduction.transform(X)
        print(f'Embeddings have been reduced to {self.n_components} dimensions successfully!!')
        return pd.DataFrame(reduced_embeddings)

class SaveEmbeddings(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.path = os.path.join(PACKAGE_ROOT, 'datasets/Train_word_embeddings.faiss')

    def change_to_test(self):
        self.path = os.path.join(PACKAGE_ROOT, 'datasets/Test_word_embeddings.faiss')

    def change_to_train(self):
        self.path = os.path.join(PACKAGE_ROOT, 'datasets/Train_word_embeddings.faiss')

    def fit(self, X, y=None):
        try:
            d = X.shape[1]
            index = faiss.IndexFlatL2(d)
            index.add(X.values.astype('float32'))
            faiss.write_index(index, self.path)
            
            print("Embeddings was saved successfully in the vector db.")
            
            return self.path
        except Exception as e:
            print(f"Error in saving embeddings in the vector db: {str(e)}")

    def transform(self, X):
        self.fit(X)
        return self

class CustomPipeline():
    def __init__(self):
        self.pipeline = Pipeline([
                    ('Gemma_2B_Embeddings', Gemma_2B_Embeddings()),
                    ('StandarScaling', StandardScaling()),
                    ('DimensionalityReduction', DimensionalityReduction()),
                    ('SaveEmbeddings', SaveEmbeddings())
        ])

