from pathlib import Path
import streamlit as st

dir_path = Path(__file__).parent


def run():
    page = st.navigation(
        [
            st.Page(
                dir_path / "home.py", title="Home", icon=":material/home:"
            ),
            st.Page(
                dir_path / "words_demo.py", title="Roget's Word Demo", icon=":material/table:"
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
    run()