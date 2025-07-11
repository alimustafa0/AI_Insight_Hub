import streamlit as st
import spacy
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import nltk.downloader
import subprocess
import os


# This directory will be created and used for NLTK downloads
NLTK_DATA_PATH = os.path.join(os.getcwd(), "nltk_data")

# This MUST be done before any nltk.data.find() or nltk.download() calls
if NLTK_DATA_PATH not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_PATH)
    
# Cache NLTK downloads and SpaCy model loading to prevent repeated execution
@st.cache_resource
def load_nltk_resources():
    """
    Downloads necessary NLTK data if not already present,
    ensuring they are stored in a persistent and accessible location.
    """
    nltk_resources = {
        'punkt': 'tokenizers/punkt',
        'punkt_tab': 'tokenizers/punkt_tab',
        'stopwords': 'corpora/stopwords',
        'wordnet': 'corpora/wordnet',
        'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger'
    }

    # Ensure the NLTK data directory exists
    os.makedirs(NLTK_DATA_PATH, exist_ok=True)

    for name, path in nltk_resources.items():
        try:
            # Try to find the resource locally within NLTK_DATA_PATH
            nltk.data.find(path, paths=[NLTK_DATA_PATH]) # Explicitly search in our path
        except LookupError:
            st.info(f"Downloading NLTK '{name}' to {NLTK_DATA_PATH}...")
            try:
                nltk.download(name, download_dir=NLTK_DATA_PATH)
            except Exception as e:
                st.error(f"Failed to download NLTK '{name}': {e}")
                return False
        except Exception as e: # Catch any other unexpected errors during find
            st.error(f"An unexpected error occurred while checking NLTK '{name}': {e}")
            return False

    return True

@st.cache_resource
def load_spacy_model():
    """
    Loads the spaCy 'en_core_web_sm' model, downloading it if not found.
    Uses st.cache_resource for efficient, one-time loading.
    """
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        st.info("SpaCy model 'en_core_web_sm' not found. Attempting to download...")
        try:
            result = subprocess.run(
                ["python", "-m", "spacy", "download", "en_core_web_sm"],
                check=True,
                capture_output=True,
                text=True
            )
            st.success(f"SpaCy model downloaded successfully: {result.stdout}")
            nlp = spacy.load("en_core_web_sm")
        except subprocess.CalledProcessError as e:
            st.error(f"Failed to download spaCy model: {e.stderr}")
            raise
        except Exception as e:
            st.error(f"An unexpected error occurred during spaCy model download or load: {e}")
            raise
    return nlp

@st.cache_resource
def load_sentiment_pipeline():
    try:
        return pipeline("sentiment-analysis")
    except Exception as e:
        st.error(f"Failed to load sentiment analysis model: {e}. Please check your internet connection and Hugging Face Transformers installation.")
        st.stop()