import streamlit as st
import os
from pathlib import Path
import pandas as pd
import numpy as np
import faiss
import plotly.express as px

dir_path = Path(__file__).parent

# Set page title and icon
st.set_page_config(page_title="Roget's Word Demo", page_icon=":material/table:")

# Title
st.title("Roget's Word Demo")

# Info
st.write("This demo illustrates information about the words and the embeddings used to train the pipeline!!")

def data_presentation():
    def get_words():
        # Load words
        words_path = os.path.join(dir_path, "data/Roget's_Words.csv")
        words = pd.read_csv(words_path, encoding='latin').loc[:, ['Class', 'Section', 'Sub-Category', 'Word']]
        words['Word'] = words['Word'].astype(str)
        
        # Remove "SECTION <RomanNumeral>. " prefix and "CLASS I " part.
        words['Section'] = words['Section'].str.replace(r'^SECTION\s+\w+\.\s*', '', regex=True).str.strip()
        words['Class'] = words['Class'].str.replace(r'^CLASS\s+\w+\s+', '', regex=True).str.strip()
        return words

    def get_embeddings(n):
        # Load embeddings
        embeddings_path = os.path.join(dir_path, f"data/embeddings_{n}d.faiss")
        index = faiss.read_index(embeddings_path)
        d = index.d
        embeddings = np.zeros((index.ntotal, d), dtype=np.float32)

        for i in range(index.ntotal):
            embeddings[i] = index.reconstruct(i)
        
        if n == 2:
            embeddings = pd.DataFrame(embeddings, columns=["X", "Y"])
        elif n == 3:
            embeddings = pd.DataFrame(embeddings, columns=["X", "Y", "Z"])
        
        return embeddings

    try:
        words, embeddings_2d, embeddings_3d = get_words(), get_embeddings(2), get_embeddings(3)

        # Search bar
        st.header("Search Words")
        search_query = st.text_input("Enter a word to search:")

        if search_query:  # Perform search only if query is not empty
            # Filter the dataframe
            filtered_df = words[words['Word'].str.contains(search_query, case=False, na=False)]
            
            if not filtered_df.empty:
                # Display results in an expander/dropdown
                with st.expander(f"Search Results ({len(filtered_df)} matches):", expanded=True):
                    st.dataframe(filtered_df[['Word', 'Sub-Category', 'Section', 'Class']], use_container_width=True)
            else:
                st.write("No results found.")

        # Class distribution
        class_distribution = words['Class'].value_counts()
        fig_class_pie = px.pie(class_distribution, 
                            values=class_distribution.values, 
                            names=class_distribution.index, 
                            title="Class Distribution")
        fig_class_pie.update_traces(
            hovertemplate="<span style='color: gray;'>Class </span> %{label}<br><span style='color: gray;'>No. Words </span> %{value}<extra></extra>",
        )
        fig_class_pie.update_layout(height=450)
        st.plotly_chart(fig_class_pie)

        # Section distribution
        section_distribution = words['Section'].value_counts()
        fig_section_pie = px.pie(section_distribution, 
                                values=section_distribution.values, 
                                names=section_distribution.index, 
                                title="Section Distribution")
        # Update hover template to customize appearance
        fig_section_pie.update_traces(
            hovertemplate="<span style='color: gray;'>Section </span> %{label}<br><span style='color: gray;'>No. Words </span> %{value}<extra></extra>",
        )
        fig_section_pie.update_layout(height=550)
        st.plotly_chart(fig_section_pie)


        # Combine embeddings with metadata for coloring
        embeddings_2d = pd.concat([embeddings_2d, words[['Class', 'Section', 'Word']].reset_index(drop=True)], axis=1)
        embeddings_3d = pd.concat([embeddings_3d, words[['Class', 'Section', 'Word']].reset_index(drop=True)], axis=1)

        # Embeddings plot header
        st.header("Embedding Plots")
        
        # User selection for hue
        hue_option = st.selectbox("Hue by:", ["Class", "Section"])
        num_samples = st.slider(
            "Select number of samples to display:",
            min_value=20000,
            max_value=len(embeddings_2d),
            value=20000,  # Default value
            step=1000
        )

        # Filter the data based on the number of samples
        filtered_embeddings_2d = embeddings_2d.sample(n=num_samples, random_state=33)
        filtered_embeddings_3d = embeddings_3d.sample(n=num_samples, random_state=33)

        # Plotly 2D Plot
        st.subheader("2D Embeddings Plot")
        fig_2d = px.scatter(
            filtered_embeddings_2d, 
            x="X", 
            y="Y", 
            color=hue_option, 
            hover_data=["Word"],  # Tooltip
        )

        # Update hover template to customize appearance
        fig_2d.update_traces(
            marker=dict(size=4, opacity=0.35),
            hovertemplate="<span style='color: gray;'>Word</span> %{customdata[0]}<extra></extra>",
            customdata=filtered_embeddings_2d[['Word']]
        )

        # Adjust legend
        fig_2d.update_layout(
            legend=dict(
                title=hue_option,
                orientation="h",  # Horizontal legend
                yanchor="bottom",  # Position the legend below the plot
                y=-0.6 if hue_option == 'Class' else -1.4,  # Move the legend further down
                xanchor="center",
                x=0.5,  # Center the legend horizontally
                traceorder="normal",
                itemsizing='constant',
            ),
            height=600 if hue_option == 'Class' else 850,  # Set height of the plot
        )
        st.plotly_chart(fig_2d, use_container_width=True)

        # Plotly 3D Plot
        st.subheader("3D Embeddings Plot")
        fig_3d = px.scatter_3d(
            filtered_embeddings_3d,
            x="X", y="Y", z="Z",
            color=hue_option,
            labels={"X": "X", "Y": "Y", "Z": "Z"},
        )

        # Update hover template to customize appearance
        fig_3d.update_traces(
            marker=dict(size=4, opacity=0.4),
            hovertemplate="<span style='color: gray;'>Word</span> %{customdata[0]}<extra></extra>",
            customdata=filtered_embeddings_3d[['Word']]
        )
        
        # Adjust legend
        fig_3d.update_layout(
            legend=dict(
                title=hue_option,
                yanchor="top",
                y=-0.05,  # Place legend below the plot
                xanchor="center",
                x=0.5,  # Center the legend
                orientation="h",  # Horizontal orientation
                traceorder="normal",
                itemsizing='constant',
            ),
            margin=dict(t=10, b=30),  # Fixed plot margins
            height=600 if hue_option == 'Class' else 930,  # Set height of the plot
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")

data_presentation()
