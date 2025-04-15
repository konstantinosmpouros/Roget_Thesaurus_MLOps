from pathlib import Path
import streamlit as st
import os
from dotenv import load_dotenv

dir_path = Path(__file__).parent


def run():
    page = st.navigation(
        [
            st.Page(
                dir_path / "home.py", title="Home", icon=":material/home:"
            ),
            st.Page(
                dir_path / "words_demo.py", title="Roget's Words Demo", icon=":material/table:"
            ),
            st.Page(
                dir_path / "predict.py", title="Predict Words", icon=':material/show_chart:',
            ),
            st.Page(
                dir_path / "about.py", title="About", icon=':material/info:',
            ),
            st.Page(
                dir_path / "contact.py", title="Contact", icon=':material/mail:',
            )
        ]
    )
    page.run()


if __name__ == "__main__":
    # Load the .env file with tokens
    load_dotenv()
    os.environ['HUGGINGFACE_TOKEN'] = os.getenv('HUGGINGFACE_TOKEN')

    run()