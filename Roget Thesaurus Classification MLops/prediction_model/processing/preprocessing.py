from sklearn.base import BaseEstimator,TransformerMixin

from pathlib import Path
import os
import sys
import pandas as pd

from umap import umap_ as umap

import torch
import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))


class Gemma7B_Embeddings(BaseEstimator, TransformerMixin):
    def __init__(self):
        model_name = "google/gemma-1.1-7b-it"
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

    def fit_transform(self, words_to_vectorize, batch_size=100):
        embeddings = []
        for start in tqdm(range(0, len(words_to_vectorize), batch_size)):
            batch = words_to_vectorize[start:start + batch_size].tolist()

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

        return pd.Dataframe(embeddings)


class DimensionalityReduction(BaseEstimator, TransformerMixin):
    def __init__(self, ):
        self.dim_reduction = umap.UMAP(n_components=300,  n_jobs=-1, random_state=33)

    def fit_transform(self, embeddings):
        return self.dim_reduction.fit_transform(embeddings)

