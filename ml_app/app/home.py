import streamlit as st

# Set page title and icon
st.set_page_config(page_title="Word Classifier", page_icon=":material/home:")

# Title
st.title("Welcome to the Word Classifier App! 📚")

# Project Overview
st.write(
    """
    This is a machine learning application designed to classify words based on their **class** and **section** 
    from Roget's Thesaurus. This project is part of an MLOps pipeline that enables easy prediction of word classifications.

    **👈 Select the prediction from the sidebar** to interact with the model or to learn more about the application.

    ### Key Features
    - **Word Classification**: Predicts the class and section of a given word.
    - **MLOps Pipeline**: Model served using FastAPI and tracked with MLflow for optimal performance.
    - **Dockerized App**: Fully containerized for easy deployment and scalability.

    ### Want to learn more about the project?
    - Explore the [GitHub repository](https://github.com/konstantinosmpouros/Roget_Thesaurus_MLOps) to see the complete code and model training pipeline.
    - Learn about the [MLOps pipeline](https://docs.your-mlops-docs) that tracks models and ensures efficient deployment.

    ### Explore the ML Model
    The app uses a pre-trained machine learning model to classify words and predict their classes and sections from **Roget's Thesaurus**.
    The model is served through a FastAPI server, and you can interact with it through the **Predictions** page.

    ### See the Pipeline in Action
    This app is designed to be used with Docker for easy deployment and scaling. You can run it on your local machine or in a cloud environment.
    """
)

# Optional: Add links or other content related to your project
st.write("Developed by [konstantinos Mpouros](https://github.com/konstantinosmpouros) as part of an MLOps learning project.")