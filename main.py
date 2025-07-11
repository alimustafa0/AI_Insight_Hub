import streamlit as st
from PIL import Image
import base64 # Import base64 for encoding

st.set_page_config(
    page_title="AI Insight Hub",
    page_icon="🤖",
    layout="wide"
)

# Function to encode image to base64
@st.cache_data # Cache the image loading to avoid re-reading on every rerun
def get_image_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        st.error(f"Error: Image file not found at {image_path}. Please ensure the 'assets' folder is in the same directory as main.py and contains banner.jpg.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the image: {e}")
        return None

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
    .stApp {
        background-color: #f5f5f5; /* Softer off-white background */
        color: #333333; /* Darker text for better readability */
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: bold;
        color: #2c3e50; /* Darker tab text */
    }
    .stExpander {
        border: 1px solid #d3d3d3; /* Lighter border */
        border-radius: 8px;
        padding: 15px; /* Slightly more padding */
        margin-bottom: 18px; /* Slightly more margin */
        background-color: #ffffff;
        box-shadow: 0 2px 5px rgba(0,0,0,0.08); /* Softer shadow */
    }
    .stExpander div[data-testid="stExpanderChevron"] {
        padding-top: 0.6rem;
        padding-bottom: 0.6rem;
    }
    h2 {
        color: #2c3e50;
        font-size: 2.4rem; /* Slightly larger heading */
        margin-bottom: 20px;
    }
    h3 {
        color: #34495e;
        font-size: 2.0rem;
        margin-bottom: 12px;
    }
    p {
        color: #444444; /* Slightly darker paragraph text */
        line-height: 1.7;
        font-size: 1.05rem;
    }
    .module-title {
        font-size: 1.6rem;
        font-weight: bold;
        color: #3498db; /* A different shade of blue */
    }
    /* Styling for the sidebar tagline */
    .sidebar-tagline {
        font-size: 1rem; /* Adjust font size */
        font-style: italic; /* Italicize */
        color: #5d6d7e; /* A slightly subdued color */
        margin-top: -10px; /* Adjust margin to bring it closer to the title */
        margin-bottom: 10px;
    }
    /* Styling for the main banner image */
    .main-banner-img {
        max-width: 100%; /* Ensure it fits within the container width */
        max-height: 400px; /* Set a maximum height to prevent it from being too tall */
        height: auto; /* Maintain aspect ratio */
        display: block; /* Ensure block level for margin auto */
        margin-left: auto; /* Center the image */
        margin-right: auto; /* Center the image */
        border-radius: 10px; /* Slightly rounded corners for the image */
        box-shadow: 0 4px 8px rgba(0,0,0,0.1); /* Subtle shadow for the image */
    }
    </style>
    """, unsafe_allow_html=True)


# --- Sidebar Content ---
st.sidebar.title("📊 AI Insight Hub")
st.sidebar.markdown("<p class='sidebar-tagline'>An advanced AI-powered data science toolkit.</p>", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.markdown("##### Developed by:")
st.sidebar.markdown("### Ali Mustafa (Sharawy)")
st.sidebar.markdown("###### Teaching Assistant at Lotus University")
st.sidebar.markdown("###### AI Engineer")
st.sidebar.markdown("---")


# --- Main Page Content ---
st.title("🚀 Welcome to AI Insight Hub")

st.markdown("""
Welcome to the **AI Insight Hub**, your go-to platform for exploring the fascinating world of Artificial Intelligence and Data Science!

This intelligent platform is designed as a comprehensive data science toolkit, seamlessly integrating advanced modules to empower your analytical and learning journey.
My goal with this platform is to provide a versatile and intuitive environment for exploring, building, and learning about AI and data science.
""")

# --- Display Banner Image with adjusted sizing and styling ---
image_base64_data = get_image_base64("assets/banner.jpg")
if image_base64_data: # Only display if image was successfully loaded
    st.markdown(
        f'<img src="data:image/jpeg;base64,{image_base64_data}" class="main-banner-img" alt="AI Insight Hub Banner">',
        unsafe_allow_html=True
    )
    st.markdown("<p style='text-align: center; color: #666666; font-style: italic; margin-top: 10px;'>Explore the Power of AI and Data Science</p>", unsafe_allow_html=True)


st.markdown("---")
st.header("Explore Our Powerful Modules:")


# --- Module 1: Machine Learning Model Building ---
with st.expander("✨ **Machine Learning Model Builder & Preprocessing**", expanded=False):
    st.markdown("<p class='module-title'>Overview</p>", unsafe_allow_html=True)
    st.markdown("<p>This module is a comprehensive web application designed to guide users through the entire machine learning pipeline, from data ingestion and preprocessing to model building and evaluation.</p>", unsafe_allow_html=True)
    st.markdown("<p class='module-title'>Key Features</p>", unsafe_allow_html=True)
    st.markdown("""
    * **Data Upload and Management:** Allows users to upload CSV files and manage data sessions effectively.
    * **Integrated Data Exploration:** Provides tools for viewing raw data, understanding data types, descriptive statistics, and correlation analysis.
    * **Interactive Data Preprocessing:** Includes options for handling missing values, feature scaling, categorical encoding, and feature selection.
    * **Machine Learning Model Building:** Supports various classification and regression tasks with a selection of popular algorithms.
    * **Model Training and Evaluation:** Facilitates data splitting, model training, and presents performance metrics with visualizations.
    """, unsafe_allow_html=True)

# --- Module 2: Natural Language Processing (NLP) ---
with st.expander("💬 **Advanced NLP Intelligence Engine**", expanded=False):
    st.markdown("<p class='module-title'>Overview</p>", unsafe_allow_html=True)
    st.markdown("<p>This module is a powerful web-based tool for extracting deep insights from text data using Natural Language Processing (NLP) and AI techniques. It supports direct text input and document uploads.</p>", unsafe_allow_html=True)
    st.markdown("<p class='module-title'>Key Features</p>", unsafe_allow_html=True)
    st.markdown("""
    * **Flexible Text Input:** Accepts direct text or uploads of `.txt` and `.pdf` files.
    * **Comprehensive Text Preprocessing:** Includes language detection, lowercasing, punctuation and stop word removal.
    * **Advanced Text Analysis:** Offers sentiment analysis, keyword extraction, text summarization, named entity recognition, and part-of-speech tagging.
    * **Visual Insights:** Generates word clouds and word frequency plots for visual understanding of text data.
    """, unsafe_allow_html=True)

# --- Module 3: Image Analysis ---
with st.expander("🖼️ **Advanced Image Classifier**", expanded=False):
    st.markdown("<p class='module-title'>Overview</p>", unsafe_allow_html=True)
    st.markdown("<p>This interactive web application is designed to classify images using deep learning models and provides functionalities for image processing and metadata exploration.</p>", unsafe_allow_html=True)
    st.markdown("<p class='module-title'>Key Features</p>", unsafe_allow_html=True)
    st.markdown("""
    * **Image Upload and Manipulation:** Allows uploading various image formats with options for brightness, contrast, sharpness adjustments, and artistic filters.
    * **Metadata Exploration:** Extracts and displays EXIF metadata from uploaded images.
    * **AI-Powered Classification:** Utilizes pre-trained deep learning models (ResNet50, MobileNetV2) for image classification with top predictions and confidence scores.
    * **Contextual Information:** Integrates with Wikipedia to provide summaries for the top predicted labels.
    """, unsafe_allow_html=True)

# --- Module 4: Interactive Data Analysis & Dashboards ---
with st.expander("📊 **Advanced Data Analyzer**", expanded=False):
    st.markdown("<p class='module-title'>Overview</p>", unsafe_allow_html=True)
    st.markdown("<p>This module is an interactive web-based tool for in-depth exploration and analysis of uploaded `.csv` datasets, providing automated insights through a user-friendly interface.</p>", unsafe_allow_html=True)
    st.markdown("<p class='module-title'>Key Features</p>", unsafe_allow_html=True)
    st.markdown("""
    * **Tab-Based Analysis:** Organizes data exploration into Overview & Cleaning, Univariate Analysis, Bivariate Analysis, and Advanced Visualizations tabs.
    * **Automated Insights:** Quickly generates summaries, statistics, and visualizations for uploaded data.
    * **Data Download:** Allows users to download the analyzed or processed datasets.
    """, unsafe_allow_html=True)

# --- Module 5: Educational Teaching Tools ---
with st.expander("🧠 **Teaching & Learning Tools**", expanded=False):
    st.markdown("<p class='module-title'>Overview</p>", unsafe_allow_html=True)
    st.markdown("<p>This comprehensive educational platform is designed to help users visualize, explain, and test fundamental concepts in Mathematics, Logic, and Programming relevant to data science.</p>", unsafe_allow_html=True)
    st.markdown("<p class='module-title'>Key Features</p>", unsafe_allow_html=True)
    st.markdown("""
    * **Interactive Visualizers:** Includes a Math Function Visualizer that plots expressions and a Logic Truth Table Generator.
    * **Code Explanation:** Provides breakdowns of Python code structures.
    * **Formula Reference:** Offers a searchable library of famous mathematical and scientific formulas.
    * **Self-Assessment:** Features a dynamic quiz generator for testing knowledge in Math, Python, and Logic.
    """, unsafe_allow_html=True)