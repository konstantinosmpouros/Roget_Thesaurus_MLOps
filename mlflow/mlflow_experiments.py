import argparse
import re

from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    auc)
from sklearn.preprocessing import label_binarize
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

import optuna
import logging

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
    'SGDClassifier': SGDClassifier,
    'SupportVectorClassifier': SVC,
    'DecisionTreeClassifier': DecisionTreeClassifier,
    'RandomForestClassifier': RandomForestClassifier,
    'XGBClassifier': XGBClassifier,
    'LGBMClassifier': LGBMClassifier,
}

# Define hyperparameter search spaces
param_spaces = {
    'SGDClassifier': {
        'alpha': lambda trial: trial.suggest_float ('alpha', 1e-5, 1e-1),
        'max_iter': lambda trial: trial.suggest_int('max_iter', 1000, 3000),
        'loss': lambda trial: trial.suggest_categorical('loss', ['log_loss']),
        'penalty': lambda trial: trial.suggest_categorical('penalty', ['l2', 'l1', 'elasticnet']),
        'l1_ratio': lambda trial: trial.suggest_float('l1_ratio', 0.0, 1.0) if trial.params.get('penalty') == 'elasticnet' else 0.0,
        'n_jobs': lambda trial : trial.suggest_categorical('n_jobs', [-1]),
    },
    
    'SupportVectorClassifier': {
        'C': lambda trial: trial.suggest_float('C', 0.1, 100.0, log=True),
        'kernel': lambda trial: trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf']),
        'gamma': lambda trial: trial.suggest_categorical('gamma', ['scale', 'auto']),
        'degree': lambda trial: trial.suggest_int('degree', 2, 5) if trial.params.get('kernel') == 'poly' else 3,
        'shrinking': lambda trial: trial.suggest_categorical('shrinking', [True, False]),
        'class_weight': lambda trial: trial.suggest_categorical('class_weight', [None, 'balanced']),
        'max_iter': lambda trial: trial.suggest_int('max_iter', 500, 1000),
        'probability': lambda trial: trial.suggest_categorical('probability', [True]),
    },
    
    'DecisionTreeClassifier': {
        'max_depth': lambda trial: trial.suggest_int('max_depth', 2, 32),
        'min_samples_split': lambda trial: trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': lambda trial: trial.suggest_int('min_samples_leaf', 1, 10),
        'criterion': lambda trial: trial.suggest_categorical('criterion', ['gini', 'entropy'])
    },
    
    'RandomForestClassifier': {
        'n_estimators': lambda trial: trial.suggest_int('n_estimators', 10, 100),
        'max_depth': lambda trial: trial.suggest_int('max_depth', 2, 32),
        'min_samples_leaf': lambda trial: trial.suggest_int('min_samples_leaf', 1, 10),
        'bootstrap': lambda trial: trial.suggest_categorical('bootstrap', [True, False]),
        'n_jobs': lambda trial : trial.suggest_categorical('n_jobs', [-1]),
    },
    
    'XGBClassifier': {
        'n_estimators': lambda trial: trial.suggest_int('n_estimators', 10, 100),
        'max_depth': lambda trial: trial.suggest_int('max_depth', 2, 10),
        'colsample_bytree': lambda trial: trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'subsample': lambda trial: trial.suggest_float('subsample', 0.5, 1.0),
        'n_jobs': lambda trial : trial.suggest_categorical('n_jobs', [-1]),
    },
    
    'LGBMClassifier': {
        'n_estimators': lambda trial: trial.suggest_int('n_estimators', 10, 100),
        'num_leaves': lambda trial: trial.suggest_int('num_leaves', 20, 100),
        'colsample_bytree': lambda trial: trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_samples': lambda trial: trial.suggest_int('min_child_samples', 5, 50),
        'max_depth': lambda trial: trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': lambda trial: trial.suggest_float('min_child_weight', 0.0, 10.0),
        'n_jobs': lambda trial : trial.suggest_categorical('n_jobs', [-1]),
        'verbose': lambda trial : trial.suggest_categorical('verbose', [-1]),
    }
}

def set_args():
    # Set up argument parser for the target parameter
    parser = argparse.ArgumentParser(description="Run CustomPipeline for word classification.")
    parser.add_argument('target', choices=['Class', 'Section'], help="Specify the target type: 'Class' or 'Section'.")
    return parser.parse_args().target

def get_labels(target, encode=False):
    # Load labels
    df = load_dataset()
    X, y = separate_data(df, target)

    # Encode labels if needed
    if encode:
        y = encode_y_data(y)

    # Split the label the same way embeddings have been splited
    _, _, y_train, y_test = split_data(X, y)
    y_train, y_test = y_train.values.ravel(), y_test.values.ravel()

    return y_train, y_test

def objective(trial, model_name, X_train, y_train, X_test, y_test):
    # Exctract model Class and Hyperparameters
    model_class = models[model_name]
    param_space = param_spaces[model_name]
    
    # Define model hyperparameters
    params = {key: lmbd(trial) for key, lmbd in param_space.items()}
    model = model_class(**params, random_state=33)
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Evaluate
    accuracy = model.score(X_test, y_test)
    
    return accuracy

def run_optuna_study(model_name, X_train, y_train, X_test, y_test, n_trials=20):
    # Mute warnings
    logging.basicConfig(level=logging.WARNING)
    
    # Initialize the study name and run it
    study_name = f"{model_name}_optimization"
    study = optuna.create_study(direction='maximize', study_name=study_name)
    study.optimize(
        lambda trial: objective(trial, model_name, X_train, y_train, X_test, y_test),
        n_trials=n_trials,
    )
    
    # Retrieve the best parameters for the current model, initialize it and train it
    best_params = study.best_trial.params
    best_model = models[model_name](**best_params, random_state=33)
    best_model.fit(X_train, y_train)
    
    return best_model

def plot_confusion_matrix(y_test, y_pred, target, model_name, fixed_labels, figsize=(15, 9)):
    # Generate the confusion matrix plot
    plt.figure(figsize=figsize)

    cm = confusion_matrix(y_test, y_pred)
    cm_sum = cm.sum(axis=1)[:, np.newaxis]

    # Calculate percentages
    cm_percentage = (cm.astype('float') / cm_sum) * 100

    # Format the text for display
    cm_percentage_text = np.array([["{:.2f}%".format(value) for value in row] for row in cm_percentage])

    # Plot the heatmap
    sns.heatmap(cm_percentage,
                annot=cm_percentage_text,
                fmt="",
                cmap="Blues",
                xticklabels=fixed_labels,
                yticklabels=fixed_labels,
                annot_kws={"size": 10},
                vmin=0,
                vmax=100
    )

    # Style the plot
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    os.makedirs("plots", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"plots/{model_name}_{target}_confusion_matrix.png")
    plt.close()

def plot_roc_auc_curve(y_test, model, X_test, target, model_name, figsize=(15, 9)):
    # Generate the roc auc curve for Multi-class problem
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    n_classes = y_test_bin.shape[1]
    y_pred_prob = np.nan_to_num(model.predict_proba(X_test), nan=0.0)

    # Calculate ROC AUC score (weighted average across all classes)
    roc_auc = roc_auc_score(y_test_bin, y_pred_prob, average='weighted', multi_class='ovr')

    # Plot ROC Curve for each class
    plt.figure(figsize=figsize)
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{target} {i} (AUC = {auc_score:.2f})")

    # Style the plot
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"{model_name} ROC Curve (Multi-class)")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize='small')
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.savefig(f"plots/{model_name}_{target}_roc_curve.png")
    plt.close()
    
    return roc_auc

def metrics_and_plots(model, X_test, y_test, model_name, target):
    # Make the predictions
    y_pred = model.predict(X_test)

    # Format-fix the labels for the plots
    labels = np.unique(y_test)

    def fix_label(labels, target):
        # Remove the Class/Section and the latin number for every category
        if target == 'Class':
            class_pattern = r"(CLASS\s+[IVXLCDM]+)"
            labels = [re.sub(class_pattern, "", str(label)).strip() for label in labels]
        elif target == 'Section':
            section_pattern = r"(SECTION\s+[IVXLCDM]+\.)"
            labels = [re.sub(section_pattern, "", str(label)).strip() for label in labels]
        return labels

    fixed_labels = fix_label(labels, target)

    # Generate the metric according to the predictions
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')

    # Plot the confusion matrix and roc auc curve for Multi-class problem
    if target == 'Class':
        plot_confusion_matrix(y_test, y_pred, target, model_name, fixed_labels)
        roc_auc = plot_roc_auc_curve(y_test, model, X_test, target, model_name)
    else:
        plot_confusion_matrix(y_test, y_pred, target, model_name, fixed_labels, figsize=(31,25))
        roc_auc = plot_roc_auc_curve(y_test, model, X_test, target, model_name)

    return accuracy, precision, f1, roc_auc

def mlflow_logging(model, model_name, target):
    with mlflow.start_run() as run:
        # Set needed tags (run id, target and the models name)
        run_id = run.info.run_id
        mlflow.set_tag("run_id", run_id)
        mlflow.set_tag("target", target)
        mlflow.set_tag('model_name', model_name)
        
        # Generate the metrics to save
        accuracy, precision, f1_score, roc_auc = metrics_and_plots(model, X_test, y_test, model_name, target)

        # Log the best parameters
        mlflow.log_params(model.get_params())

        # Log the metrics
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("F1_score", f1_score)
        mlflow.log_metric("roc_auc_score", roc_auc)

        # Logging model and plots
        mlflow.log_artifact(f"plots/{model_name}_{target}_confusion_matrix.png")
        mlflow.log_artifact(f"plots/{model_name}_{target}_roc_curve.png")
        mlflow.sklearn.log_model(model, model_name)

        # End the run
        mlflow.end_run()


if __name__ == '__main__':
    # Set the CLI parameters
    target = set_args()

    # Set the url and the experiment name for mlflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    # mlflow.set_tracking_uri("http://127.0.0.1:5001") # Use this if you want to store the logs in a mysql database
    mlflow.set_experiment("Roget_Classification")

    # Load embeddings
    X_train, X_test = load_embeddings()
    
    print(f'\nStarting the {target} project!!\n')

    for model_name in models.keys():
        # Get the data (Train + test labels)
        if model_name in ['XGBClassifier']:
            y_train, y_test = get_labels(target, encode=True)
        else:
            y_train, y_test = get_labels(target)

        # Train and evaluate a non optimized model
        print(f'Non optimized {model_name}...')

        model = models[model_name](random_state=33)
        model.fit(X_train, y_train)
        
        score_1 = model.score(X_test, y_test)
        print(f'Non optimized {model_name} accuracy: {score_1}\n')

        # Train an optuna optimized model
        print(f"Optimizing {model_name}...")
        best_trial = run_optuna_study(model_name,
                                      X_train,
                                      y_train,
                                      X_test,
                                      y_test)
        score_2 = best_trial.score(X_test, y_test)
        print(f'Best optuna optimized {model_name} accuracy: {score_2}')

        # Save parameters, model and tags for this model
        print(f'Logging best {model_name} model and parameters')
        mlflow_logging(best_trial, model_name, target)
        print('Model and parameters successfully saved!!\n\n')
