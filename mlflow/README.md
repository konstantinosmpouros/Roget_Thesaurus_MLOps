# Roget Thesaurus Classification MLOps Project - Mlflow experimentation

This project is an end-to-end MLOps pipeline designed to classify words based on their semantic categories in Roget's Thesaurus.
The pipeline loads data previously extracted from [Roget's Thesaurus](https://www.gutenberg.org/cache/epub/22/pg22-images.html), generates embeddings using the Gemma 1.1 2B it model, performs dimensionality reduction and Standard Scaling, and uses a LightGBM model to predict the semantic category or section for each word.

## Mlflow project

This mlflow project is dedicated in the optimazation of the machine learning models that will predict the Class or the Section of the given word embeddings. The process is divided in two parts. First is the embeddings generation where a pipeline is trained on train set to generate the word embeddings using the last hidden state of the Gemma 1.1 2B model and also predicts the test set embeddings. This process is ilustrated in the directories, data_processing and datasets. When the embeddings are generated and stored in vector dbs then comes the mlflow traching and experimantation. The mlflow_experimantation coducts 2 experimentations, depending on the parameters given on the execution. One is the Class and the other is the Section experimentation. In its experementation a set of machine learning models are optimized using the optuna and stored (parameters and model instances) in the the Roget_Classification experiment.

## Run the mlflow project

To run the mlflow project you can use the run_project.sh script. Be aware that before running the project you need to go to the data_processing directory and run the embeddings_generation.py. The requirements to run this python script and the mlflow project are in the requirement file. Please be sure that you have installed in you os the pyenv cause is need to run the mlflow project.

To run the embeddings_generation.py on your virtual enviroment:

```bash
pip install -r requirements.txt
```

```bash
python3 embeddings_generation.py
```

To run the project since you have already generated the embeddings use these command for the Class and the Section:

```bash
mlflow run . --experiment-name Roget_Classification
```

```bash
mlflow run . --experiment-name Roget_Classification -P target=Section
```

To run a local server with one of the trained models are a rest api use the command and replace the tag with the models id from the mlflow ui artifacts:

```bash
mlflow models serve -m runs:/<RUN_ID>/model --port 9000
```

The url to communicate with the rest api is the following and an exaple body for a request can be found in the Postman_body.txt:

```text
http://127.0.0.1:9000/invocations
```
