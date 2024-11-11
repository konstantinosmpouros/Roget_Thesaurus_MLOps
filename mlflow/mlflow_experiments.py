import argparse

from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import optuna
import logging

from sklearn.metrics import accuracy_score

import mlflow
import os
import sys
from pathlib import Path

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(str(PACKAGE_ROOT))

from data_processing.data_handling import load_dataset, load_embeddings, separate_data
from data_processing.data_handling import split_data, encode_y_data


# Define models
models = {
    # 'SGDClassifier': SGDClassifier,
    # 'DecisionTreeClassifier': DecisionTreeClassifier,
    # 'RandomForestClassifier': RandomForestClassifier,
    # 'BaggingClassifier': BaggingClassifier,
    # 'GradientBoostingClassifier': GradientBoostingClassifier,
    'XGBClassifier': XGBClassifier,
    'LGBMClassifier': LGBMClassifier,
}

# Define hyperparameter search spaces
param_spaces = {
    'SGDClassifier': {
        'alpha': lambda trial: trial.suggest_float ('alpha', 1e-5, 1e-1),
        'max_iter': lambda trial: trial.suggest_int('max_iter', 1000, 3000)
    },
    'DecisionTreeClassifier': {
        'max_depth': lambda trial: trial.suggest_int('max_depth', 2, 32),
        'min_samples_split': lambda trial: trial.suggest_int('min_samples_split', 2, 20)
    },
    'RandomForestClassifier': {
        'n_estimators': lambda trial: trial.suggest_int('n_estimators', 10, 200),
        'max_depth': lambda trial: trial.suggest_int('max_depth', 2, 32)
    },
    'BaggingClassifier': {
        'n_estimators': lambda trial: trial.suggest_int('n_estimators', 10, 200),
        'max_samples': lambda trial: trial.suggest_float('max_samples', 0.1, 1.0)
    },
    'GradientBoostingClassifier': {
        'n_estimators': lambda trial: trial.suggest_int('n_estimators', 10, 200),
        'learning_rate': lambda trial: trial.suggest_float ('learning_rate', 0.01, 0.5),
        'max_depth': lambda trial: trial.suggest_int('max_depth', 2, 32)
    },
    'XGBClassifier': {
        'n_estimators': lambda trial: trial.suggest_int('n_estimators', 10, 200),
        'learning_rate': lambda trial: trial.suggest_float ('learning_rate', 0.01, 0.5),
        'max_depth': lambda trial: trial.suggest_int('max_depth', 2, 32)
    },
    'LGBMClassifier': {
        'n_estimators': lambda trial: trial.suggest_int('n_estimators', 10, 200),
        'learning_rate': lambda trial: trial.suggest_float ('learning_rate', 0.01, 0.5),
        'num_leaves': lambda trial: trial.suggest_int('num_leaves', 20, 200)
    }
}

def set_args():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run CustomPipeline for word classification.")
    parser.add_argument('target', choices=['Class', 'Section'], help="Specify the target type: 'Class' or 'Section'.")
    return parser.parse_args().target

def get_data(encode=False):
    df = load_dataset()
    _, y = separate_data(df, target)
    X = load_embeddings()
    
    if encode:
        y = encode_y_data(y)

    return split_data(X, y)

def objective(trial, model_name, X_train, y_train, X_test, y_test):
    model_class = models[model_name]
    param_space = param_spaces[model_name]
    
    # Define model hyperparameters
    params = {key: lmbd(trial) for key, lmbd in param_space.items()}
    model = model_class(**params, random_state=33)
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

def run_optuna_study(model_name, X_train, y_train, X_test, y_test, n_trials=50):
    logging.basicConfig(level=logging.WARNING)
    
    study_name = f"{model_name}_optimization"
    study = optuna.create_study(direction='maximize', study_name=study_name)
    study.optimize(
        lambda trial: objective(trial, model_name, X_train, y_train, X_test, y_test),
        n_trials=n_trials,
    )
    
    best_params = study.best_trial.params
    best_model = models[model_name](**best_params)
    best_model.fit(X_train, y_train)
    
    return best_model, accuracy_score(y_test, best_model.predict(X_test))

def mlflow_logging(model, accuracy, model_name, target):
    mlflow.set_experiment("Roget_Classification")
    
    with mlflow.start_run() as run:
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        run_id = run.info.run_id
        mlflow.set_tag("run_id", run_id)
        mlflow.set_tag("target", target)
        
        # Logging best parameters
        mlflow.log_params(model.get_params())
        
        #log the metrics
        mlflow.log_metric("Accuracy", accuracy)
        
        # Logging model
        mlflow.sklearn.log_model(model, model_name)

if __name__ == '__main__':
    target = set_args()

    for model_name in models.keys():
        if model_name in ['XGBClassifier']:
            X_train, X_test, y_train, y_test = get_data(encode=True)
        else:
            X_train, X_test, y_train, y_test = get_data()
        
        # Train a non optimized model
        print(f'Non optimized {model_name}...')
        model = models[model_name](random_state=33)
        model.fit(X_train, y_train.values.ravel())
        accuracy_1 = accuracy_score(y_test, model.predict(X_test))
        print(f'Non optimized {model_name} accuracy: {accuracy_1}')
        
        # Train an optuna optimized model
        print(f"Optimizing {model_name}...")
        best_trial, accuracy_2 = run_optuna_study(model_name,
                                                X_train,
                                                y_train.values.ravel(),
                                                X_test,
                                                y_test.values.ravel())
        print(f'Best optuna optimized {model_name} accuracy: {accuracy_2}')
        
        print(f'Logging best optuna {model_name} parameters')
        mlflow_logging(best_trial, accuracy_2, model_name, target)
        print('Model and parameters successfully saved!!\n\n')
        