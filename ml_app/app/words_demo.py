import streamlit as st
import os
from pathlib import Path
import pandas as pd
import numpy as np
import faiss

dir_path = Path(__file__).parent

# Set page title and icon
st.set_page_config(page_title="Roget's Word Demo", page_icon=":material/table:")

# Title
st.title("Roget's Word Demo")

# Info
st.write("This demo is to illustrate the information about the words and the embeddings used to train the pipeline!!")

def data_presentation():
    def get_data():
        # Load words
        words_path = os.path.join(dir_path, "data/Roget's_Words.csv")
        words = pd.read_csv(words_path, encoding='latin').loc[:, ['Class', 'Section', 'Word']]
        
        # Load 2D embeddings
        embeddings_2d_path = os.path.join(dir_path, "data/embeddings_2d.faiss")
        index = faiss.read_index(embeddings_2d_path)
        d = index.d
        embeddings_2d = np.zeros((index.ntotal, d), dtype=np.float32)

        for i in range(index.ntotal):
            embeddings_2d[i] = index.reconstruct(i)
        embeddings_2d = pd.DataFrame(embeddings_2d)
        
        # Load 3D embeddings
        embeddings_3d_path = os.path.join(dir_path, "data/embeddings_3d.faiss")
        index = faiss.read_index(embeddings_3d_path)
        d = index.d
        embeddings_3d = np.zeros((index.ntotal, d), dtype=np.float32)

        for i in range(index.ntotal):
            embeddings_3d[i] = index.reconstruct(i)
        embeddings_3d = pd.DataFrame(embeddings_3d)
        
        return words, embeddings_2d, embeddings_3d

    try:
        words, embeddings_2d, embeddings_3d = get_data()
    
    except Exception as e:
        st.error(f"Error: {e.args}")

data_presentation()
