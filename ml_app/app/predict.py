import requests
import streamlit as st

MODEL_SERVER_URL = "http://model_server:8081/predict/"

st.set_page_config(page_title="Predict Words", page_icon=":material/show_chart:")

st.title("Word Classifier")
st.write("Enter a word to get its class and section.")

# Input form
word = st.text_input("Word:")

if st.button("Submit"):
    if word.strip():
        try:
            # Send the word to the model server
            response = requests.post(MODEL_SERVER_URL, json={"word": word})
            if response.status_code == 200:
                data = response.json()
                st.success(f"Prediction:\nClass - {data['class']}, Section - {data['section']}")
            else:
                st.error("Error communicating with model server.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a valid word.")
