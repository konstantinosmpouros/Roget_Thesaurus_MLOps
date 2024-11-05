# Roget Thesaurus Classification MLOps Project (UNDERDEVELOPMENT)

This project is an end-to-end MLOps pipeline designed to classify words based on their semantic categories in Roget's Thesaurus. The pipeline extracts data from Roget's Thesaurus, generates embeddings using the Gemma 1.1 7B it model, performs dimensionality reduction, and uses an XGBoost model to predict the semantic category or section for each word. The solution is packaged as a set of APIs to streamline embedding generation and classification tasks.

## Project Overview

1. **Embedding Generation**: Generate embeddings for each word using the Gemma 1.1 7B model.
2. **Dimensionality Reduction**: Apply dimensionality reduction to the embeddings to improve computational efficiency.
3. **Classification Model**: Train an XGBoost model to classify words into semantic categories based on the reduced embeddings.
4. **APIs**: Deploy REST APIs to generate embeddings for new words, perform classification, and retrieve results.








