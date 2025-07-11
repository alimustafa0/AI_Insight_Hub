import streamlit as st
import spacy
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from transformers import pipeline

# Cache NLTK downloads and SpaCy model loading to prevent repeated execution
@st.cache_resource
def load_nltk_resources():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    except Exception as e:
        st.error(f"Failed to download NLTK resources: {e}. Please check your internet connection.")
    return True

@st.cache_resource
def load_spacy_model():
    try:
        # Ensure 'en_core_web_sm' is downloaded: python -m spacy download en_core_web_sm
        return spacy.load("en_core_web_sm")
    except Exception as e:
        st.error(f"Failed to load SpaCy model 'en_core_web_sm': {e}. Please run `python -m spacy download en_core_web_sm` in your terminal.")
        st.stop() # Stop the app if the model can't be loaded
    return None # Should not be reached if st.stop() is called

@st.cache_resource
def load_sentiment_pipeline():
    try:
        return pipeline("sentiment-analysis")
    except Exception as e:
        st.error(f"Failed to load sentiment analysis model: {e}. Please check your internet connection and Hugging Face Transformers installation.")
        st.stop()