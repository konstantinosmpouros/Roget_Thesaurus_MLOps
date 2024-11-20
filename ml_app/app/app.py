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
                dir_path / "predict.py",
                title="Predict Words",
                icon=':material/show_chart:',
            ),
        ]
    )
    page.run()


if __name__ == "__main__":
    run()