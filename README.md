# 🤖 AI Insight Hub
**An Advanced AI-Powered Data Science Toolkit**

---

## 🚀 Project Overview

Welcome to the **AI Insight Hub**! This intelligent platform is your go-to resource for exploring the fascinating world of Artificial Intelligence and Data Science. Developed as a comprehensive toolkit, it seamlessly integrates advanced modules to empower your analytical and learning journey. Whether you're a student, researcher, or practitioner, the AI Insight Hub provides a versatile and intuitive environment for exploring, building, and learning about cutting-edge AI and data science concepts.

![AI Insight Hub Banner](assets/banner.jpg)

## ✨ Key Features & Modules

The AI Insight Hub is composed of several powerful, interconnected modules, each designed to address specific data science challenges:

### 💡 Machine Learning Model Builder & Preprocessing
This module is a comprehensive web application designed to guide users through the entire machine learning pipeline, from data ingestion and preprocessing to model building and evaluation.

* **Data Upload and Management:** Allows users to upload CSV files and manage data sessions effectively.
* **Integrated Data Exploration:** Provides tools for viewing raw data, understanding data types, descriptive statistics, and correlation analysis.
* **Interactive Data Preprocessing:** Includes options for handling missing values, feature scaling, categorical encoding, and feature selection.
* **Machine Learning Model Building:** Supports various classification and regression tasks with a selection of popular algorithms.
* **Model Training and Evaluation:** Facilitates data splitting, model training, and presents performance metrics with visualizations.

### 💬 Advanced NLP Intelligence Engine
This module is a powerful web-based tool for extracting deep insights from text data using Natural Language Processing (NLP) and AI techniques. It supports direct text input and document uploads.

* **Flexible Text Input:** Accepts direct text or uploads of `.txt`, `.pdf`, and `.docx` files.
* **Comprehensive Text Preprocessing:** Includes language detection, lowercasing, punctuation and stop word removal.
* **Advanced Text Analysis:** Offers sentiment analysis, keyword extraction, text summarization, named entity recognition, and part-of-speech tagging.
* **Visual Insights:** Generates word clouds and word frequency plots for visual understanding of text data.

### 🖼️ Advanced Image Classifier
This interactive web application is designed to classify images using deep learning models and provides functionalities for image processing and metadata exploration.

* **Image Upload and Manipulation:** Allows uploading various image formats with options for brightness, contrast, sharpness adjustments, and artistic filters.
* **Metadata Exploration:** Extracts and displays EXIF metadata from uploaded images.
* **AI-Powered Classification:** Utilizes pre-trained deep learning models (ResNet50, MobileNetV2) for image classification with top predictions and confidence scores.
* **Contextual Information:** Integrates with Wikipedia to provide summaries for the top predicted labels.

### 📊 Advanced Data Analyzer (Interactive Dashboards)
This module is an interactive web-based tool for in-depth exploration and analysis of uploaded `.csv` datasets, providing automated insights through a user-friendly interface.

* **Purpose:** Provides automated insights, profiling, and visualizations for uploaded datasets.
* **Input:** Takes a `.csv` file as input.
* **Core Functionality (Tabs):** Organizes data analysis into five distinct tabs:
    * **Overview & Cleaning:** Initial data inspection, quality checks, and potential cleaning options.
    * **Univariate Analysis:** Analysis of single variables (e.g., distributions, statistics).
    * **Bivariate Analysis:** Analysis of relationships between two variables.
    * **Advanced Visualizations:** More complex or customizable plotting capabilities.
    * **Download Data:** Allows downloading processed or analyzed data.

### 🧠 Teaching & Learning Tools
This comprehensive educational platform is designed to help users visualize, explain, and test fundamental concepts in Mathematics, Logic, and Programming relevant to data science.

* **Interactive Visualizers:** Includes a Math Function Visualizer that plots expressions and a Logic Truth Table Generator.
* **Code Explanation:** Provides breakdowns of Python code structures.
* **Formula Reference:** Offers a searchable library of famous mathematical and scientific formulas.
* **Self-Assessment:** Features a dynamic quiz generator for testing knowledge in Math, Python, and Logic.

## 🛠️ Technologies Used

This project is built primarily with Python and leverages the following key libraries and frameworks:

* **Python:** The core programming language.
* **Streamlit:** For creating interactive web applications and dashboards.
* **Pandas:** For data manipulation and analysis.
* **NumPy:** For numerical operations.
* **Matplotlib, Plotly:** For data visualization.
* **Scikit-learn:** For machine learning algorithms and preprocessing.
* **PyTorch, torchvision:** For deep learning models (e.g., image classification).
* **NLTK, spaCy, TextBlob, Rake-NLTK, Sumy:** For Natural Language Processing tasks.
* **SymPy:** For symbolic mathematics and function plotting.
* **Pillow (PIL):** For image processing.
* **Wikipedia-API:** For fetching contextual information.
* **Piexif:** For EXIF metadata handling.

## ⚙️ Installation

To set up and run the AI Insight Hub locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/alimustafa0/AI_Insight_Hub.git
    cd ai_insight_hub
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    * **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

    *Note: Ensure your `requirements.txt` file is up-to-date with all necessary packages. You can generate it using `pip freeze > requirements.txt` if you have all dependencies installed in your current environment.*

5.  **Download NLTK resources (if not already handled by `nlp_engine.py`):**
    Some NLP functionalities might require NLTK data. If you encounter errors, run the following in your Python environment:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    ```
    And for spaCy, you might need to download a model:
    ```bash
    python -m spacy download en_core_web_sm
    ```

## ▶️ Usage

Once the installation is complete, you can run the Streamlit application:

1.  **Activate your virtual environment** (if not already active).
2.  **Run the Streamlit app:**
    ```bash
    streamlit run main.py
    ```
3.  Your browser will automatically open to the AI Insight Hub application (usually at `http://localhost:8501`).

Interact with the various modules by navigating through the expanders on the main page.

## 📂 Project Structure

The project is organized into a modular and clean structure:

```bash
ai_insight_hub/
├── .streamlit/             # Streamlit configuration files (e.g., config.toml)
├── assets/                 # Static assets like images (banner.jpg) and Lottie animations
│   ├── lottie/             # Lottie animation JSON files
│   └── banner.jpg          # Main banner image for the application
├── modules/                # Core logic for each data science functionality
│   ├── image_classifier.py # Logic for image classification and processing
│   ├── ml_trainer.py       # Machine learning model training and preprocessing logic
│   ├── data_analyzer.py    # Utilities for data analysis and visualization
│   ├── nlp_engine.py       # Core functions for Natural Language Processing
│   └── utils.py            # General utility functions (math, logic, quiz generation, etc.)
├── pages/                  # Streamlit pages for each module (if multi-page app structure is used)
│   ├── __init__.py         # Python package initializer
│   ├── Data_Analyzer.py    # Streamlit app page for the Data Analyzer module
│   ├── Image_Classifier.py # Streamlit app page for the Image Classifier module
│   ├── Learning_Tools.py   # Streamlit app page for the Teaching & Learning Tools module
│   ├── ML_Model_Builder.py # Streamlit app page for the ML Model Builder module
│   └── NLP_Analyzer.py     # Streamlit app page for the NLP Analyzer module
├── venv/                   # Python virtual environment (contains installed packages)
├── main.py                 # The main Streamlit application entry point
├── requirements.txt        # List of Python dependencies for environment setup
└── README.md               # Project documentation and overview
```

## 🧑‍💻 Developer

**Ali Mustafa (Sharawy)**

* Teaching Assistant at Lotus University
* AI Engineer

Feel free to connect or reach out for any inquiries!
