import pandas as pd
import requests
import streamlit as st
from streamlit_searchbox import st_searchbox

FASTAPI_URL = "http://fastapi_server:8081"

def get_words():
    # Fetches words from the API
    return pd.DataFrame(requests.get(f"{FASTAPI_URL}/get_words").json()['words'])

# Function to filter words for the searchbox
def search_words(searchterm: str) -> list:
    # Load the dataset
    words = get_words()
    matches = words[words['Word'].str.contains(searchterm, case=False, na=False)]
    return matches['Word'].tolist() if not matches.empty else []

# Function to send a request to the model server
def send_request(word):
    response = requests.post(f'{FASTAPI_URL}/predict', json={"word": word})
    if response.status_code == 200:
        return response.json()  # Assumes the response is a JSON with predictions
    else:
        return None

# Word predict main
def predict():
    # Use the searchbox to select a word from the dataset
    selected_word = st_searchbox(
        search_words,
        placeholder="Search for a word...",
        key="search_box",
    )

    # Predict button
    if st.button("Predict"):
        if selected_word:
            result = send_request(selected_word)
            if result:
                class_pred = result.get('class', 'N/A')
                predicted_section = result.get("section", "N/A")
                
                st.success(f"The predicted class is: {class_pred['0']}")
                st.success(f"The predicted section is: {predicted_section['0']}")
            else:
                st.error("Error occurred while fetching predictions.")
        else:
            st.info("Please select a word first!")


# Set page configs
st.set_page_config(page_title="Predict Words", page_icon=":material/show_chart:")

st.title("Word Classifier")
st.write("Let's get some prediction together and check the performance of the models!!")

predict()
