import pytest

from pathlib import Path
import os
import sys
import pandas as pd
import numpy as np

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config
from prediction_model.processing.preprocessing import Gemma_2B_Embeddings, StandardScaling
from prediction_model.pipeline import CustomPipeline


pipelines = [CustomPipeline(config.TARGET_CLASS), CustomPipeline(config.TARGET_SECTION)]

@pytest.fixture
def gemma2b_layer():
    return Gemma_2B_Embeddings()


# Test the gemma model is loaded properly
def test_load_gemma2b(gemma2b_layer):
    tokenizer, model = gemma2b_layer.load_model()

    assert gemma2b_layer.model_name == "google/gemma-1.1-2b-it", 'Model name must be the path google/gemma-1.1-2b-it'
    assert tokenizer is not None, "Tokenizer should be loaded"
    assert model is not None, "Model should be loaded"
    assert model.config.model_type == "gemma", "The model should be of type gemma (since it's from the gemma family)"


# Test the case where empty input is passed to the model
def test_empty_input(gemma2b_layer):
    empty_data = pd.DataFrame({0: []})

    embeddings_df = gemma2b_layer.transform(empty_data)

    # Ensure an empty DataFrame is returned
    assert embeddings_df.empty, "Embeddings should be empty for empty input data"
    assert isinstance(embeddings_df, pd.DataFrame), "Embeddings should be returned as a DataFrame"


# Test with invalid input (e.g., non-text input)
def test_invalid_input(gemma2b_layer):
    invalid_data = pd.DataFrame({0: [123, 456, 789]})  # Non-text data

    # Expecting the transform to fail and raise a ValueError
    with pytest.raises(ValueError) as ex:
        embeddings_df = gemma2b_layer.transform(invalid_data)
        assert embeddings_df.shape[0] == len(invalid_data)

    assert str(ex.value) == "text input must be of type `str` (single example), `List[str]` (batch or single pretokenized example) or `List[List[str]]` (batch of pretokenized examples)."


# Test the transform of a small valid dataset
def test_transform_small(gemma2b_layer):
    sample_data = pd.DataFrame({0: ["This is a test sentence.",
                                    "Gemma2B embeddings are useful for NLP tasks.",
                                    "Another example sentence."]})
    
    embeddings_df = gemma2b_layer.transform(sample_data)

    assert isinstance(embeddings_df, pd.DataFrame), "Embeddings should be returned as a DataFrame"
    assert embeddings_df.shape[1] == 2048, f"Expected embedding size of 2024, but got {embeddings_df.shape[1]}"
    assert embeddings_df.shape[0] == len(sample_data), "The number of embeddings should match the number of input sentences"


# Test the transform of a larger valid dataset
def test_transform_large(gemma2b_layer):
    sample_data = pd.DataFrame({0: ["Sentence {}".format(i) for i in range(2000)]})
    
    embeddings_df = gemma2b_layer.transform(sample_data)

    assert isinstance(embeddings_df, pd.DataFrame), "Embeddings should be returned as a DataFrame"
    assert embeddings_df.shape[1] == 2048, f"Expected embedding size of 2024, but got {embeddings_df.shape[1]}"
    assert embeddings_df.shape[0] == len(sample_data), "The number of embeddings should match the number of input sentences"


def test_standard_scaling_basic():
    data = pd.DataFrame(np.random.rand(10, 5) * 100)  # Random data for scaling
    scaler = StandardScaling()
    
    scaled_data = scaler.fit_transform(data)
    assert np.allclose(scaled_data.mean(), 0, atol=1e-4), "Mean should be approximately 0"
    assert np.allclose(scaled_data.std(axis=0), 1, atol=1e-1), "Standard deviation should be approximately 1"
    assert scaled_data.shape == data.shape, "Shape should remain the same after scaling"

