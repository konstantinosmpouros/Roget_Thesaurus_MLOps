# Roget Thesaurus Classification MLOps Project - Mlflow experimentation

This project is an end-to-end MLOps pipeline designed to classify words based on their semantic categories in Roget's Thesaurus.
The pipeline loads data previously extracted from [Roget's Thesaurus](https://www.gutenberg.org/cache/epub/22/pg22-images.html), generates embeddings using the Gemma 1.1 2B it model, performs dimensionality reduction and Standard Scaling, and uses a LightGBM model to predict the semantic category or section for each word.

## Mlflow project

This mlflow project is dedicated in the optimazation of the machine learning models that will predict the Class or the Section of the given word embeddings. The process is divided in two parts. First is the embeddings generation where a pipeline is trained on train set to generate the word embeddings using the last hidden state of the Gemma 1.1 2B model and also predicts the test set embeddings. This process is ilustrated in the directories, data_processing and datasets. When the embeddings are generated and stored in vector dbs then comes the mlflow traching and experimantation. The mlflow_experimantation coducts 2 experimentations, depending on the parameters given on the execution. One is the Class and the other is the Section experimentation. In its experementation a set of machine learning models are optimized using the optuna and stored (parameters and model instances) in the the Roget_Classification mlflow experiment.

## Run the mlflow project

To run the mlflow project you can use the run_project.sh script. The requirements to run generate the embeddings and the mlflow project are in the requirement file and in the python_env yaml file. The mlflow will automatically create a python virtual enviroment to run the project inside it, please be sure that you have installed in your os the pyenv cause is need from the mlflow to create the python virtual enviroment. Before running anything make sure you have runned in a separate CLI the mlflow server (mlflow ui) and then to run the project use these commands:

To run the project locally use these command for the Class and the Section:

```bash
mlflow run . --experiment-name Roget_Classification
```

```bash
mlflow run . --experiment-name Roget_Classification -P target=Section
```

To run remote from Github the Mlflow project run these commands for the Class and the Section experiments.

```bash
mlflow run https://github.com/konstantinosmpouros/Roget_Thesaurus_MLOps.git#mlflow --experiment-name Roget_Classification
```

```bash
mlflow run https://github.com/konstantinosmpouros/Roget_Thesaurus_MLOps.git#mlflow --experiment-name Roget_Classification -P target=Section
```

To run a local server with one of the trained models as a rest api use this command and replace the tag with the models id from the mlflow ui and the port tag with the port you want. I used the 9000:

```bash
mlflow models serve -m runs:/<RUN_ID>/model --port <port>
```

The url to communicate with the rest api is the following, replace the port with the port you used, and an exaple body for a request can be found in the Postman_body.txt:

```text
http://127.0.0.1:<port>/invocations
```

To store the loggs in a mysql database and in the mlruns directory run the mlflow server with this command

```bash
mlflow server --host 127.0.0.1 --port 5001 --backend-store-uri mysql://{username}:{password}@localhost/{database_name} --default-artifact-root $PWD/mlruns
```
