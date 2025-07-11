import streamlit as st
import pandas as pd
import numpy as np

# Import all necessary functions from ml_trainer.py
from modules.ml_trainer import (
    load_data,
    display_preprocessing_section,
    display_model_building_section,
    display_all_data_tabs
)

# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="ML Data Analyzer & Preprocessing")

st.title("🚀 ML Data Analyzer and Preprocessing Tool")

# --- Session State Initialization ---
# Initialize all session state variables that might hold persistent data
if 'df' not in st.session_state:
    st.session_state.df = None
if 'original_df' not in st.session_state:
    st.session_state.original_df = None
if 'uploaded_file_content' not in st.session_state:
    st.session_state.uploaded_file_content = None # To detect if the same file is re-uploaded
if 'scaled_columns' not in st.session_state:
    st.session_state.scaled_columns = []
if 'encoded_columns' not in st.session_state:
    st.session_state.encoded_columns = []
if 'selected_features_for_model' not in st.session_state:
    st.session_state.selected_features_for_model = []
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = None
if 'X_columns' not in st.session_state:
    st.session_state.X_columns = None
if 'target_column' not in st.session_state: # Ensure target_column is initialized
    st.session_state.target_column = None
if 'selected_features' not in st.session_state: # Ensure selected_features is initialized
    st.session_state.selected_features = []
if 'selected_model_name' not in st.session_state: # Ensure model selection is initialized
    st.session_state.selected_model_name = None
if 'selected_model_params' not in st.session_state: # Ensure model params are initialized
    st.session_state.selected_model_params = {}
if 'test_size' not in st.session_state: # Ensure test_size is initialized
    st.session_state.test_size = 0.2
if 'prev_problem_type' not in st.session_state: # Ensure prev_problem_type is initialized
    st.session_state.prev_problem_type = None

# Initialize the session state variable for the active main tab label
if 'active_main_tab_label' not in st.session_state:
    st.session_state.active_main_tab_label = "📊 Data Exploration & Analysis" # Default to the first tab's label


# --- Data Upload Section ---
st.header("📂 Upload Your Dataset")

# Custom CSS for tab-like buttons and file uploader
st.markdown("""
    <style>
    /* General button styling for tabs */
    .stButton>button {
        width: 100%;
        min-width: 320px; /* Increased min-width for uniform sizing, adjusted for longest text */
        border-radius: 0.75rem; /* Rounded corners */
        border: none; /* No default border */
        background: linear-gradient(145deg, #f0f2f6, #ffffff); /* Light, subtle gradient */
        color: #555; /* Softer dark gray text */
        padding: 0.9rem 1.3rem; /* Slightly more padding */
        font-size: 1.05rem;
        font-weight: 600;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1); /* Smoother, more natural transitions */
        margin: 0 0.4rem; /* Increased margin for better separation */
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08); /* Soft, diffused shadow */
        cursor: pointer;
        position: relative;
        overflow: hidden;
        text-align: center;
        letter-spacing: 0.02em; /* Slightly increased letter spacing */
        text-transform: uppercase; /* Uppercase for a modern look */
    }

    /* Hover effect for tabs */
    .stButton>button:hover {
        background: linear-gradient(145deg, #e6e9ed, #f9fbfd); /* Darker subtle gradient on hover */
        color: #333; /* Darker text on hover */
        transform: translateY(-4px); /* More noticeable lift effect */
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15); /* More pronounced shadow */
    }

    /* Active/Clicked state for tabs */
    .stButton>button:active {
        background: linear-gradient(45deg, #3cb371, #2e8b57); /* Darker green gradient when active */
        color: white;
        border-color: #2e8b57;
        box-shadow: inset 0 3px 8px rgba(0, 0, 0, 0.3); /* Deeper inset shadow for pressed look */
        transform: translateY(0); /* Reset lift */
    }

    /* Style for the currently active tab button (persisted state) */
    .stButton>button.active-tab {
        background: linear-gradient(45deg, #4CAF50, #66BB6A); /* Vibrant green gradient */
        color: white;
        border: 1px solid #4CAF50; /* Solid green border for active */
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2); /* Maintained shadow for active */
        transform: translateY(-2px); /* Slight lift for active to show prominence */
        font-weight: 700; /* Bolder text for active */
    }

    /* Underline effect for active tab */
    .stButton>button.active-tab::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100%; /* Full width underline */
        height: 5px; /* Thicker underline */
        background-color: #1e90ff; /* Dodger Blue for a clean highlight */
        border-radius: 0 0 0.75rem 0.75rem; /* Rounded bottom corners */
        animation: slideIn 0.5s forwards; /* Animation for the underline */
    }

    @keyframes slideIn {
        from {
            width: 0%;
        }
        to {
            width: 100%;
        }
    }

    /* Adjust column spacing for a tighter tab bar */
    div[data-testid="stColumns"] {
        gap: 1rem; /* Increased gap for better visual breathing room */
        margin-bottom: 2.5rem; /* More space below the tab bar */
        justify-content: center; /* Center the tab buttons */
    }

    /* Ensure the button text is centered within the button */
    .stButton>button div {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100%;
    }

    /* --- File Uploader Styling --- */
    /* Main dropzone container */
    div[data-testid="stFileUploaderDropzone"] {
        border: 3px dashed #4DB6AC; /* Thicker, vibrant dashed border (teal) */
        border-radius: 2rem; /* Even more rounded */
        background: linear-gradient(135deg, #F8FAFC, #E0F2F1); /* Soft teal-ish gradient */
        padding: 4rem; /* Even more padding for a grander feel */
        transition: all 0.5s cubic-bezier(0.23, 1, 0.32, 1); /* Smoother, more dramatic transitions */
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15); /* Deeper, softer shadow */
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 250px; /* Increased min-height */
        text-align: center;
        position: relative; /* For pseudo-elements */
        overflow: hidden; /* To contain any overflow from animations */
    }

    /* Pseudo-element for a subtle background animation */
    div[data-testid="stFileUploaderDropzone"]::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at center, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0) 70%);
        transform: rotate(45deg);
        animation: pulseEffect 8s infinite alternate ease-in-out;
        opacity: 0.5;
        z-index: 0; /* Behind content */
    }

    @keyframes pulseEffect {
        0% { transform: scale(1) rotate(45deg); opacity: 0.5; }
        100% { transform: scale(1.1) rotate(47deg); opacity: 0.7; }
    }

    /* Hover effect for dropzone */
    div[data-testid="stFileUploaderDropzone"]:hover {
        border-color: #00796B; /* Darker teal on hover */
        background: linear-gradient(135deg, #E0F2F1, #F8FAFC); /* Inverted gradient on hover */
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.25); /* Even more pronounced shadow */
        transform: translateY(-8px); /* More dramatic lift */
    }

    /* Label text for file uploader */
    label[data-testid="stFileUploaderLabel"] {
        font-size: 1.8rem; /* Much larger font size */
        font-weight: 900; /* Black bold */
        color: #00796B; /* Deep teal */
        margin-bottom: 2rem; /* More space below label */
        text-shadow: 2px 2px 4px rgba(0,0,0,0.05); /* Subtle text shadow */
        z-index: 1; /* Ensure text is above pseudo-element */
    }

    /* Instructions text inside the dropzone */
    div[data-testid="stFileUploaderDropzoneInstructions"] > div > span {
        font-size: 1.3rem; /* Larger instructions */
        color: #424242; /* Dark gray for instructions */
        font-weight: 600;
        line-height: 1.8; /* Increased line height for elegance */
        margin-bottom: 2rem; /* More space below instructions */
        z-index: 1;
    }

    /* "Browse files" button inside the dropzone */
    div[data-testid="stFileUploaderDropzoneInstructions"] button {
        background: linear-gradient(45deg, #26A69A, #00897B); /* Teal gradient */
        color: white;
        border-radius: 1rem; /* Very rounded button */
        padding: 1rem 2.5rem; /* More padding */
        font-weight: bold;
        transition: all 0.4s cubic-bezier(0.23, 1, 0.32, 1);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.25); /* Stronger shadow for button */
        margin-top: 2rem; /* More space above button */
        border: none;
        text-transform: uppercase;
        letter-spacing: 0.1em; /* More letter spacing for button text */
        font-size: 1.2rem; /* Larger font size for button */
        z-index: 1;
    }

    /* Hover effect for "Browse files" button */
    div[data-testid="stFileUploaderDropzoneInstructions"] button:hover {
        background: linear-gradient(45deg, #00897B, #26A69A); /* Darker teal gradient */
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35);
        transform: translateY(-5px); /* More noticeable lift */
    }

    /* Active effect for "Browse files" button */
    div[data-testid="stFileUploaderDropzoneInstructions"] button:active {
        box-shadow: inset 0 4px 10px rgba(0, 0, 0, 0.5); /* Deeper inset shadow */
        transform: translateY(0);
    }

    /* File name display after upload */
    div[data-testid="stFileUploaderFileName"] {
        font-size: 1.2rem; /* Larger file name */
        color: #00796B; /* Dark teal text */
        font-weight: 700;
        margin-top: 2.5rem; /* More space above file name */
        padding: 1rem 2rem; /* More padding */
        background-color: #E0F2F1; /* Light teal background */
        border-radius: 1rem; /* More rounded */
        border: 2px solid #80CBC4; /* Softer teal border */
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.15); /* More prominent shadow */
        z-index: 1;
    }

    /* Clear button for file uploader */
    div[data-testid="stFileUploaderClearButton"] button {
        background-color: #e74c3c; /* Red for clear button */
        color: white;
        border-radius: 0.75rem;
        padding: 0.5rem 1.2rem;
        font-size: 1rem;
        font-weight: bold;
        transition: all 0.3s ease-in-out;
        border: none;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    div[data-testid="stFileUploaderClearButton"] button:hover {
        background-color: #c0392b; /* Darker red on hover */
        transform: translateY(-2px);
    }
    </style>
""", unsafe_allow_html=True)


uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"], key="main_file_uploader")

# Logic to handle file upload and state reset
if uploaded_file is not None:
    # Read the content of the uploaded file
    current_file_content = uploaded_file.getvalue()

    # Check if a new file has been uploaded or if the content has changed
    if st.session_state.uploaded_file_content is None or current_file_content != st.session_state.uploaded_file_content:
        # Load data using the cached function from ml_trainer.py
        new_df = load_data(uploaded_file)
        if new_df is not None:
            st.session_state.df = new_df.copy()
            st.session_state.original_df = new_df.copy() # Store original for reset
            st.session_state.uploaded_file_content = current_file_content # Store content to detect changes

            # Reset ALL relevant states when a new file is uploaded
            st.session_state.scaled_columns = []
            st.session_state.encoded_columns = []
            st.session_state.selected_features_for_model = []
            st.session_state.trained_model = None
            st.session_state.X_test = None
            st.session_state.y_test = None
            st.session_state.problem_type = None
            st.session_state.X_columns = None
            st.session_state.target_column = None
            st.session_state.selected_features = []
            st.session_state.selected_model_name = None
            st.session_state.selected_model_params = {}
            st.session_state.test_size = 0.2
            st.session_state.prev_problem_type = None
            st.session_state.active_main_tab_label = "📊 Data Exploration & Analysis" # Reset to first tab on new upload

            st.success("✅ File uploaded successfully! Data loaded and states reset.")
            st.rerun() # Rerun to ensure a clean state and update UI
        else:
            # If load_data returns None (error in reading file)
            st.session_state.df = None
            st.session_state.original_df = None
            st.session_state.uploaded_file_content = None # Clear content on error
            # Reset other states as well
            st.session_state.scaled_columns = []
            st.session_state.encoded_columns = []
            st.session_state.selected_features_for_model = []
            st.session_state.trained_model = None
            st.session_state.X_test = None
            st.session_state.y_test = None
            st.session_state.problem_type = None
            st.session_state.X_columns = None
            st.session_state.target_column = None
            st.session_state.selected_features = []
            st.session_state.selected_model_name = None
            st.session_state.selected_model_params = {}
            st.session_state.test_size = 0.2
            st.session_state.prev_problem_type = None
            st.session_state.active_main_tab_label = "📊 Data Exploration & Analysis"
            st.rerun() # Rerun to clear any invalid state and update UI
    else:
        st.info("Same file detected. Using the already loaded dataset.")
elif st.session_state.df is not None:
    # This block executes if no file is currently uploaded in the widget,
    # but there's still data in session_state.df (e.g., after a browser refresh).
    # This ensures the app clears previous data if the user explicitly removes the file.
    st.session_state.df = None
    st.session_state.original_df = None
    st.session_state.uploaded_file_content = None
    # Reset other states as well
    st.session_state.scaled_columns = []
    st.session_state.encoded_columns = []
    st.session_state.selected_features_for_model = []
    st.session_state.trained_model = None
    st.session_state.X_test = None
    st.session_state.y_test = None
    st.session_state.problem_type = None
    st.session_state.X_columns = None
    st.session_state.target_column = None
    st.session_state.selected_features = []
    st.session_state.selected_model_name = None
    st.session_state.selected_model_params = {}
    st.session_state.test_size = 0.2
    st.session_state.prev_problem_type = None
    st.session_state.active_main_tab_label = "📊 Data Exploration & Analysis"
    st.info("No file uploaded. Previous data and results cleared for a fresh start.")
    st.rerun() # Rerun to ensure the UI updates to the empty state


# The rest of your app's logic will only execute if df is not None.
# If df is None, it will fall to the final 'else' block prompting for upload.
if st.session_state.df is not None:
    df = st.session_state.df # Use the dataframe from session state for current operations

    # Define the titles for the main tabs
    main_tab_titles = [
        "📊 Data Exploration & Analysis",
        "⚙️ Data Preprocessing",
        "🧠 Model Building & Evaluation"
    ]

    # Create columns for the custom tab buttons
    # Using explicit column widths to ensure equal distribution
    cols = st.columns([1, 1, 1]) # Distribute space equally among 3 columns
    
    for i, tab_title in enumerate(main_tab_titles):
        with cols[i]:
            button_key = f"tab_button_{i}"
            
            is_active = (st.session_state.active_main_tab_label == tab_title)
            
            if st.button(tab_title, key=button_key):
                st.session_state.active_main_tab_label = tab_title
                st.rerun() # Rerun to update the active tab styling and content

            # After rendering the button, apply the active-tab class using markdown
            if is_active:
                st.markdown(f"""
                    <script>
                        var buttons = window.parent.document.querySelectorAll('.stButton button');
                        buttons.forEach(function(button) {{
                            // Check if the button's text content matches the active tab title
                            // Using textContent for more reliable matching
                            if (button.textContent.trim() === '{tab_title.strip()}') {{
                                button.classList.add('active-tab');
                            }} else {{
                                button.classList.remove('active-tab');
                            }}
                        }});
                    </script>
                """, unsafe_allow_html=True)


    # Use if/elif/else to display content based on the selected tab
    if st.session_state.active_main_tab_label == main_tab_titles[0]:
        display_all_data_tabs(st.session_state.df)
    elif st.session_state.active_main_tab_label == main_tab_titles[1]:
        display_preprocessing_section(st.session_state.df)
    elif st.session_state.active_main_tab_label == main_tab_titles[2]:
        display_model_building_section(st.session_state.df)

else:
    st.info("Waiting for dataset upload...")
