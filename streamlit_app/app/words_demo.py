import streamlit as st
import os
from pathlib import Path
import pandas as pd
import numpy as np
import faiss
import altair as alt
import plotly.express as px

dir_path = Path(__file__).parent

# Set page title and icon
st.set_page_config(page_title="Roget's Word Demo", page_icon=":material/table:")

# Title
st.title("Roget's Word Demo")

# Info
st.write("This demo illustrates information about the words and the embeddings used to train the pipeline!!")

def data_presentation():
    def get_data():
        # Load words
        words_path = os.path.join(dir_path, "data/Roget's_Words.csv")
        words = pd.read_csv(words_path, encoding='latin').loc[:, ['Class', 'Section', 'Word']]
        # Cleaning
        words['Word'] = words['Word'].astype(str)
        words['Section'] = words['Section'].str.replace(r'^SECTION\s+\w+\.\s*', '', regex=True).str.strip()  # Remove "SECTION <RomanNumeral>. " prefix
        words['Class'] = words['Class'].str.replace(r'^CLASS\s+\w+\s+', '', regex=True).str.strip()  # Remove "CLASS I " part

        # Load 2D embeddings
        embeddings_2d_path = os.path.join(dir_path, "data/embeddings_2d.faiss")
        index = faiss.read_index(embeddings_2d_path)
        d = index.d
        embeddings_2d = np.zeros((index.ntotal, d), dtype=np.float32)

        for i in range(index.ntotal):
            embeddings_2d[i] = index.reconstruct(i)
        embeddings_2d = pd.DataFrame(embeddings_2d, columns=["X", "Y"])

        # Load 3D embeddings
        embeddings_3d_path = os.path.join(dir_path, "data/embeddings_3d.faiss")
        index = faiss.read_index(embeddings_3d_path)
        d = index.d
        embeddings_3d = np.zeros((index.ntotal, d), dtype=np.float32)

        for i in range(index.ntotal):
            embeddings_3d[i] = index.reconstruct(i)
        embeddings_3d = pd.DataFrame(embeddings_3d, columns=["X", "Y", "Z"])

        return words, embeddings_2d, embeddings_3d

    try:
        words, embeddings_2d, embeddings_3d = get_data()

        # Groupby Class
        class_word_counts = words.groupby('Class')['Word'].count().reset_index()
        class_word_counts.rename(columns={'Word': 'No. Words'}, inplace=True)
        class_word_counts.set_index('Class', inplace=True)

        # Groupby Section
        section_word_counts = words.groupby('Section')['Word'].count().reset_index()
        section_word_counts.rename(columns={'Word': 'No. Words'}, inplace=True)
        section_word_counts.set_index('Section', inplace=True)

        # Display dataframes
        st.dataframe(class_word_counts.T)
        st.dataframe(section_word_counts.T)

        # Combine embeddings with metadata for coloring
        embeddings_2d = pd.concat([embeddings_2d, words[['Class', 'Section', 'Word']].reset_index(drop=True)], axis=1)
        embeddings_3d = pd.concat([embeddings_3d, words[['Class', 'Section', 'Word']].reset_index(drop=True)], axis=1)

        # User selection for hue
        st.subheader("Choose how to hue the plots:")
        hue_option = st.selectbox("Hue by:", ["Class", "Section"])

        # Altair 2D Plot
        st.subheader("2D Plot of Embeddings")
        chart_2d = alt.Chart(embeddings_2d.reset_index()).mark_circle(size=60).encode(
            x=alt.X("X", scale=alt.Scale(zero=False)),
            y=alt.Y("Y", scale=alt.Scale(zero=False)),
            color=alt.Color(hue_option, legend=alt.Legend(title=hue_option)),
            tooltip=["Word"],
            opacity=alt.value(0.4)
        ).properties(
            width=600,
            height=400,
            title=f"2D Embeddings (Hue by {hue_option})"
        ).interactive()
        st.altair_chart(chart_2d, use_container_width=True)

        # Plotly 3D Plot
        st.subheader("3D Plot of Embeddings")
        
        fig_3d = px.scatter_3d(
            embeddings_3d,
            x="X", y="Y", z="Z",
            color=hue_option,
            title=f"3D Embeddings (Hue by {hue_option})",
            labels={"X": "X", "Y": "Y", "Z": "Z"},
            opacity=0.5
        )
        
        fig_3d.update_traces(marker=dict(size=5))
        
        # Update hover template to customize appearance
        fig_3d.update_traces(
            hovertemplate="<span style='color: gray;'>Word</span> %{customdata[0]}<extra></extra>",
            customdata=embeddings_3d[['Word']]
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e.args}")

data_presentation()
