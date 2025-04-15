import streamlit as st
import base64
from pathlib import Path

# Set page title and icon
st.set_page_config(page_title="Roget's Word Classifier", page_icon=":material/home:")

# Encode the local image in base64
def load_image_base64(image_path):
    img_path = Path(image_path)
    with open(img_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return f"data:image/png;base64,{encoded}"

# Load the local icon
icon_base64 = load_image_base64("materials/ai_icon_gemma.png")

# Title with local AI icon
st.markdown(
    f"""
    <h1 style="display: inline; font-size: 3em; margin-bottom: 20px;">
        Welcome to the Roget's Word Classifier App! 
        <img src="{icon_base64}" alt="AI Icon" style="width: 1.8em; height: auto; margin-left: 0px; vertical-align: left;">
    </h1>
    """,
    unsafe_allow_html=True,
)

# Additional spacing between the title and the text
st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

# Intro to the Concept of the Project
st.write(
    """
    This application is designed to classify words from **Roget's Thesaurus** into their respective **class** and **section**. 
    It leverages an MLOps pipeline to ensure a streamlined process for prediction, tracking, and deployment.
    
    The project focuses on automating the classification process, allowing users to input any word, and receive its predicted 
    class and section. This is powered by a pre-trained machine learning model that has been optimized and deployed using modern MLOps tools.
    """
)
