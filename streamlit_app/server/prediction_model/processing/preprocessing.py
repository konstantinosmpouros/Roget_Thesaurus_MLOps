from sklearn.base import BaseEstimator,TransformerMixin

from pathlib import Path
import os
import sys
import pandas as pd

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
# from huggingface_hub import login

from sklearn.preprocessing import StandardScaler
from umap import UMAP

import optuna
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))


class Gemma_2B_Embeddings(BaseEstimator, TransformerMixin):
    def __init__(self):
        # self.hf_login()
        
        # Set the model name for Gemma 1.1 2B model
        self.model_name = "google/gemma-1.1-2b-it"

        # Load the tokenizer corresponding to the model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load the pre-trained model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map='auto',
            torch_dtype=torch.bfloat16, 
        )

        # Set the model to evaluation mode
        self.model.eval()

    # Method to set the random seed for reproducibility across runs
    def set_seed(self):
        torch.manual_seed(33)
        torch.cuda.manual_seed_all(33)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # def hf_login(self):
    #     # Load Hugging Face token from environment variables
    #     hf_token = os.getenv("HUGGINGFACE_TOKEN")
    #     if not hf_token:
    #         raise ValueError("Hugging Face token not found in environment variables. Set the 'HUGGINGFACE_TOKEN' variable.")
    #     else:
    #         print('HF Token successfully found!!')
                
    #     # Log in to Hugging Face using the token
    #     login(token=hf_token)

    # The fit method is a placeholder since no fitting is required for this transformer
    def fit(self, X, y=None):
        return self

    def transform(self, X, batch_size=100):
        # Set the seed for reproducibility
        self.set_seed()
        embeddings = [] # List to store the embeddings

        # Process the input data in batches for memory efficiency
        for start in tqdm(range(0, len(X), batch_size)):
            # Select a batch of text data
            batch = X.iloc[start:start + batch_size, 0].tolist() 

            # Tokenize the batch of text
            batch_tokenized  = self.tokenizer(batch,
                                         truncation=True,
                                         padding='max_length',
                                         max_length=20,
                                         return_tensors='pt').to('cuda')

            # Perform inference without updating gradients
            with torch.no_grad():
                outputs = self.model(**batch_tokenized, output_hidden_states=True)

            # Extract the last hidden states from the model output
            last_hidden_states = outputs.hidden_states[-1]
            batch_word_embedding  = last_hidden_states.mean(dim=1)
            # Append the computed embeddings to the list
            embeddings.extend(batch_word_embedding.cpu().float().numpy())
        
        print('Embeddings have been created successfully!!')
        return pd.DataFrame(embeddings)

class StandardScaling(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Initialize the StandardScaler to standardize the data
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        # Fit the scaler to the data (calculating the mean and std)
        self.scaler.fit(X)
        return self

    def transform(self, X):
         # Apply the learned scaling transformation to the input data
        scaled_x = self.scaler.transform(X)
        
        print('Embeddings have been Standar Scaled successfully!!')
        return pd.DataFrame(scaled_x)

class DimensionalityReduction(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Set the number of components to reduce the embeddings to (default 300)
        self.n_components = 300
        
        # Initialize UMAP as the dimensionality reduction algorithm
        self.dim_reduction = UMAP(n_components=self.n_components, random_state=33)

    def fit(self, X, y=None):
        # Fit the UMAP model to the data
        self.dim_reduction.fit(X)
        return self

    def transform(self, X):
        # Apply the learned dimensionality reduction to the X data
        reduced_embeddings =  self.dim_reduction.transform(X)
        
        print(f'Embeddings have been reduced to {self.n_components} dimensions successfully!!')
        return pd.DataFrame(reduced_embeddings)

class LGBM(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.model = None
        self.best_params = None
        self.study = None
    
    def fit(self, X, y=None):
        # Convert y to 1D
        y = y.squeeze()
        
        # Initialize the objective function
        def objective(trial):
            params = {
                'num_leaves': trial.suggest_int('num_leaves', 20, 70),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'max_depth': trial.suggest_int('max_depth', 3, 6),
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-3, 10.0),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 10.0),
                'min_split_gain': trial.suggest_float('min_split_gain', 1e-3, 0.1),
                'n_estimators': trial.suggest_int('n_estimators', 50, 100),
                'verbose': -1,
                'n_jobs': -1,
            }

            # Initialize the model with parameters
            model = LGBMClassifier(**params)
            
            # Perform cross-validation with 3 folds
            cross_val_scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
            
            return cross_val_scores.mean()

        # Perform hyperparameter optimization with Optuna
        self.study = optuna.create_study(direction='maximize')
        self.study.optimize(objective, n_trials=20)
        
        # Store the best parameters and fit the final model
        self.best_params = self.study.best_params
        self.model = LGBMClassifier(**self.best_params)
        self.model.fit(X, y.squeeze())
        
        print('LGBM with the best parameters successfully trained!!')
        
        return self

    def predict(self, X):
        if self.model is not None:
            # Predict using the trained model
            predictions = self.model.predict(X)
            # Return the predictions
            return pd.Series(predictions)
        else:
            raise ValueError("Model has not been fitted yet. Call `fit` first.")
    
    def score(self, X, y):
        if self.model is not None:
            # Predict using the trained model
            y_pred = self.model.predict(X)
            # Return the accuracy score of the model
            return accuracy_score(y, y_pred)
        else:
            raise ValueError("Model has not been fitted yet. Call `fit` first.")
