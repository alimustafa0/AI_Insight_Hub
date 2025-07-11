import streamlit as st
from modules.data_analyzer import load_data, display_tab1, display_tab2, display_tab3, display_tab4, display_tab5

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Advanced Data Analyzer")
st.title("📈 Advanced Data Analyzer")
st.markdown("Upload a `.csv` file and get automatic insights, profiling, and visualizations for your dataset.")

# --- Data Upload and Caching ---


uploaded_file = st.file_uploader("Upload your dataset (CSV only)", type=["csv"])

# Load data and store in session state if not already present or if a new file is uploaded
if uploaded_file is not None:
    # Check if a new file is uploaded or if the session state df is from a different file
    if 'last_uploaded_file_id' not in st.session_state or st.session_state.last_uploaded_file_id != uploaded_file.file_id:
        st.session_state.df = load_data(uploaded_file)
        st.session_state.last_uploaded_file_id = uploaded_file.file_id # Store an identifier for the current file
        if st.session_state.df is not None:
            st.success("✅ File successfully uploaded and loaded!")
    else:
        # If the same file is uploaded again, and df is in session state, don't reload from scratch
        # This prevents re-loading the original uncached version unless a new file is detected.
        if 'df' not in st.session_state or st.session_state.df is None:
             st.session_state.df = load_data(uploaded_file)
             if st.session_state.df is not None:
                 st.success("✅ File successfully uploaded and loaded!")
else:
    # Clear session state df if no file is uploaded
    if 'df' in st.session_state:
        del st.session_state['df']
    if 'last_uploaded_file_id' in st.session_state:
        del st.session_state['last_uploaded_file_id']


# Check if a DataFrame is available in session state
if 'df' not in st.session_state or st.session_state.df is None:
    st.info("Please upload a dataset to begin the analysis.")
    st.stop() # Stop execution if no valid file is uploaded or read

df_current = st.session_state.df # Always work with the DataFrame from session state


# --- Tabs for Organized Analysis Sections ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview & Cleaning", 
    "Univariate Analysis", 
    "Bivariate Analysis", 
    "Advanced Visualizations", 
    "Download Data"
])

with tab1: # Overview & Cleaning
    display_tab1(df_current)

with tab2: # Univariate Analysis
    display_tab2(df_current)

with tab3: # Bivariate Analysis
    display_tab3(df_current)

with tab4: # Advanced Visualizations
    display_tab4(df_current)

with tab5: # Download Data
    display_tab5(df_current)