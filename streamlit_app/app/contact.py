import streamlit as st

# Set page title and icon
st.set_page_config(page_title="Contact", page_icon=":material/mail:")

# Contact Page
st.title("Contact")

st.write(
    """
    This project was developed by [Konstantinos Mpouros](https://www.linkedin.com/in/konstantinos-mpouros-5b491219b/), 
    as part of an MLOps learning initiative. 
    
    The goal was to apply machine learning and modern MLOps practices to the classification of words, 
    while ensuring scalability and reproducibility in production environments.
    """
)

st.write(
    """
    ### Developer Information
    - **Name**: Konstantinos Mpouros
    - **Email**: kostasbouros@hotmail.gr
    - **LinkedIn**: [Konstantinos Mpouros](https://www.linkedin.com/in/konstantinos-mpouros-5b491219b/)
    - **GitHub**: [konstantinosmpouros](https://github.com/konstantinosmpouros/)
    - **Project Repository**: [Roget_Thesaurus_MLOps](https://github.com/konstantinosmpouros/Roget_Thesaurus_MLOps)
    """
)

st.write("Feel free to reach out for questions, feedback, or collaborations!")