# Roget Thesaurus Classification MLOps Project - STREAMLIT WEB APP

This directory hosts an interactive web application for classifying words into their semantic categories based on Roget's Thesaurus. The web app is built on a client-server architecture, with the client using Streamlit for the UI and the server using FastAPI for handling predictions and data processing.

For detailed information about the underlying pipeline, please visit the test_pypi directory in the main project.

## Web App Features

1. **Client-Server Architecture**:

    - Client: Built with Streamlit for a user-friendly interface.

    - Server: Powered by FastAPI, managing data and executing predictions entirely on the server-side.

2. **Word Classification**:

    - Input any word of the Roget's Thesaurus and see its actual and predicted categories.

3. **Server-Side Processing**:

    - All data availability, embedding computations, and predictions are executed on the server for efficiency and scalability.

4. **Embeddings Visualization**:

    - Optional feature to explore words in a reduced-dimensionality space.

5. **Dockerized Deployment**:

    - The project is available as a Docker image, simplifying the deployment process.
