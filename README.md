# Roget Thesaurus Classification MLOps Project (UNDERDEVELOPMENT)

This project is an end-to-end MLOps pipeline designed to classify words based on their semantic categories in Roget's Thesaurus.
The pipeline loads data previously extracted from [Roget's Thesaurus](https://www.gutenberg.org/cache/epub/22/pg22-images.html), generates embeddings using the Gemma 1.1 2B it model, performs dimensionality reduction and Standard Scaling, and uses a LightGBM model to predict the semantic category or section for each word.

## Pipeline Overview

The pipeline consists of the following key stages:

1. **Embedding Generation**: Word embeddings are generated using the Gemma 1.1 2B model, which converts words into dense vector representations based on their semantic meanings.

2. **Standard Scaling**: The embeddings are standardized to ensure that the features have a mean of 0 and a standard deviation of 1, helping improve the performance of downstream models.

3. **Dimensionality Reduction**: We apply UMAP dimensionality reduction techniques to reduce the complexity of the embeddings.

4. **Standard Scaling**: The embeddings are standardized again after the dimensionality reduction to ensure that the features have a mean of 0 and a standard deviation of 1, helping improve the performance of the classification models.

5. **Classification**: An LightGBM classifier is trained to predict the semantic category (or section) for each word, based on the processed embeddings.

## TestPyPI Project and Installation

The above mentioned pipeline and the code is also available as a python library through TestPyPI by running the following command in the CLI:

```bash
pip install -i https://test.pypi.org/simple/ roget-thesaurus-classification
```

To run train and predict with the pipeline, make sure you have the necessary libraries by running the command with the downloaded requirements file:

```bash
pip install -r requirements.txt
```

## TestPyPI Project Structure

![alt text](images/image.png)

## Mlflow Project

The mlflow project is dedicated in the optimazation of the machine learning models that will predict the Class or the Section of the given word embeddings. The process is divided in two parts. First is the embeddings generation where a pipeline is trained on train set to generate the word embeddings using the last hidden state of the Gemma 1.1 2B model and also predicts the test set embeddings. This process is ilustrated in the directories, data_processing and datasets. When the embeddings are generated and stored in vector dbs then comes the mlflow traching and experimantation. The mlflow_experimantation coducts 2 experimentations, depending on the parameters given on the execution. One is the Class and the other is the Section experimentation. In its experementation a set of machine learning models are optimized using the optuna and stored (parameters and model instances) in the the Roget_Classification experiment.

## Mlflow Project Structure

![alt text](images/image1.png)

## TODO

MLApps
Jenkins
fastapi
docker
