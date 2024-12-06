from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import numpy as np
import pandas as pd
import faiss
import sys
import os
from pathlib import Path

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.pipeline import CustomPipeline
from prediction_model.config import config

app = FastAPI()

# Load pipelines
class_pipeline = CustomPipeline(config.TARGET_CLASS)
class_pipeline.load_pipeline()

section_pipeline = CustomPipeline(config.TARGET_SECTION)
section_pipeline.load_pipeline()

# Perform word parsing
class WordPred(BaseModel):
    word: str

# Perform embeddings dim parsing
class Embeddings(BaseModel):
    dimensions: int

@app.post('/predict')
def predict(word_details: WordPred):
    # Parse the word and pass it into a dataframe
    df = pd.DataFrame({'Word': [word_details.model_dump()['word']]})

    # Make the predictions
    class_pred = class_pipeline.pipeline.predict(df)
    section_pred = section_pipeline.pipeline.predict(df)

    # Return the predictions in a dictionary
    predictions = {
        'class': class_pred,
        'section': section_pred
    }
    return predictions

@app.get('/get_words')
def get_words():
    words_path = os.path.join(PACKAGE_ROOT, "data/Roget's_Words.csv")
    
    # Load the CSV and clean column names
    df = pd.read_csv(words_path, encoding='latin')
    df.columns = df.columns.str.strip()  # Remove extra spaces
    
    # Select necessary columns (adjust as per actual CSV structure)
    words = df.loc[:, ['Class', 'Section', 'Sub-Category', 'Word']]
    words['Word'] = words['Word'].astype(str)

    # Clean 'Section' and 'Class' columns
    words['Section'] = words['Section'].str.replace(r'^SECTION\s+\w+\.\s*', '', regex=True).str.strip()
    words['Class'] = words['Class'].str.replace(r'^CLASS\s+\w+\s+', '', regex=True).str.strip()

    # Convert to JSON-serializable format
    words_dict = words.to_dict(orient='records')
    
    return {"words": words_dict}

@app.post('/get_embeddings')
def get_embeddings(embed_parser: Embeddings):
    # Parse the dimensions number
    n = embed_parser.model_dump()['dimensions']

    # Load embeddings if they exist with the current dim number
    embeddings_path = os.path.join(PACKAGE_ROOT, f"data/embeddings_{n}d.faiss")

    # If the file exists try to load the embeddings
    if os.path.exists(embeddings_path):
        # Read the FAISS index
        index = faiss.read_index(embeddings_path)
        d = index.d
        embeddings = np.zeros((index.ntotal, d), dtype=np.float32)

        # Reconstruct embeddings from the index
        for i in range(index.ntotal):
            embeddings[i] = index.reconstruct(i)

        # Create a DataFrame with appropriate column names
        if n == 2:
            embeddings = pd.DataFrame(embeddings, columns=["X", "Y"])
        elif n == 3:
            embeddings = pd.DataFrame(embeddings, columns=["X", "Y", "Z"])
        else:
            raise HTTPException(status_code=400, detail="Unsupported dimensions.")

        # Convert DataFrame to a list of dictionaries (JSON-serializable format)
        return embeddings.apply(lambda col: col.map(float)).to_dict(orient="records")

    else:
        raise HTTPException(status_code=404, detail="Embeddings do not exist for this number of dimensions.")

if __name__ == '__main__':
	uvicorn.run(app, host='127.0.0.1', port=8081)