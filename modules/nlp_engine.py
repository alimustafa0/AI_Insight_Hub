import streamlit as st
import spacy
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import nltk.downloader
import subprocess

# Cache NLTK downloads and SpaCy model loading to prevent repeated execution
@st.cache_resource
def load_nltk_resources():
    """
    Downloads necessary NLTK data if not already present.
    Uses st.cache_resource to ensure data is downloaded only once per deployment.
    """
    # Define the NLTK resources needed
    nltk_resources = {
        'punkt': 'tokenizers/punkt',
        'stopwords': 'corpora/stopwords',
        'wordnet': 'corpora/wordnet',
        'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger'
    }

    for name, path in nltk_resources.items():
        try:
            # Try to find the resource locally
            nltk.data.find(path)
        except LookupError: # <--- CORRECTED: Catch LookupError here
            st.info(f"Downloading NLTK '{name}'...")
            try:
                nltk.download(name)
            except Exception as e: # Catch errors during the actual download
                st.error(f"Failed to download NLTK '{name}': {e}")
                return False # Indicate failure
        except Exception as e: # Catch any other unexpected errors during find
            st.error(f"An unexpected error occurred while checking NLTK '{name}': {e}")
            return False # Indicate failure

    return True # Indicate success

@st.cache_resource
def load_spacy_model():
    """
    Loads the spaCy 'en_core_web_sm' model, downloading it if not found.
    Uses st.cache_resource for efficient, one-time loading.
    """
    try:
        # Attempt to load the model
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        st.info("SpaCy model 'en_core_web_sm' not found. Attempting to download...")
        try:
            # Use subprocess to run the spaCy download command
            # This is more robust for deployment environments
            result = subprocess.run(
                ["python", "-m", "spacy", "download", "en_core_web_sm"],
                check=True, # Raise CalledProcessError if command returns non-zero exit status
                capture_output=True, # Capture stdout and stderr
                text=True # Decode stdout/stderr as text
            )
            st.success(f"SpaCy model downloaded successfully: {result.stdout}")
            nlp = spacy.load("en_core_web_sm") # Load the model after successful download
        except subprocess.CalledProcessError as e:
            st.error(f"Failed to download spaCy model: {e.stderr}")
            raise # Re-raise the exception to indicate a critical failure
        except Exception as e:
            st.error(f"An unexpected error occurred during spaCy model download or load: {e}")
            raise # Re-raise other exceptions
    return nlp

@st.cache_resource
def load_sentiment_pipeline():
    try:
        return pipeline("sentiment-analysis")
    except Exception as e:
        st.error(f"Failed to load sentiment analysis model: {e}. Please check your internet connection and Hugging Face Transformers installation.")
        st.stop()