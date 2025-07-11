import streamlit as st
import spacy
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import nltk.downloader

# Cache NLTK downloads and SpaCy model loading to prevent repeated execution
@st.cache_resource
def load_nltk_resources():
    """
    Downloads necessary NLTK data if not already present.
    Uses st.cache_resource to ensure data is downloaded only once per deployment.
    """
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError: # This line will now work correctly
        st.info("Downloading NLTK 'punkt' tokenizer...")
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        st.info("Downloading NLTK 'stopwords' corpus...")
        nltk.download('stopwords')

    try:
        nltk.data.find('corpora/wordnet')
    except nltk.downloader.DownloadError:
        st.info("Downloading NLTK 'wordnet' corpus...")
        nltk.download('wordnet')

    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except nltk.downloader.DownloadError:
        st.info("Downloading NLTK 'averaged_perceptron_tagger'...")
        nltk.download('averaged_perceptron_tagger')

    return True # Indicate success

@st.cache_resource
def load_spacy_model():
    """
    Loads the spaCy 'en_core_web_sm' model, downloading it if not found.
    Uses st.cache_resource for efficient, one-time loading.
    """
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        st.info("SpaCy model 'en_core_web_sm' not found. Downloading...")
        # Use spacy.cli.download to download the model programmatically
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm") # Load after download
    return nlp

@st.cache_resource
def load_sentiment_pipeline():
    try:
        return pipeline("sentiment-analysis")
    except Exception as e:
        st.error(f"Failed to load sentiment analysis model: {e}. Please check your internet connection and Hugging Face Transformers installation.")
        st.stop()