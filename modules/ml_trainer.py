import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import joblib # For saving/loading models
import io # For in-memory binary streams for model download

# Import necessary scikit-learn modules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Import models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC

# Try to import XGBoost, provide a warning if not installed
try:
    from xgboost import XGBRegressor, XGBClassifier
except ImportError:
    st.warning("XGBoost not installed. Please install it (`pip install xgboost`) to use XGBoost models.")
    XGBRegressor = None
    XGBClassifier = None


# --- Helper Functions (assuming these are already in ml_trainer.py) ---

@st.cache_data
def load_data(uploaded_file_obj):
    """Loads CSV data and caches it."""
    if uploaded_file_obj is not None:
        try:
            df = pd.read_csv(uploaded_file_obj)
            return df
        except Exception as e:
            st.error(f"Error reading CSV file: {e}. Please ensure it's a valid CSV.")
            return None
    return None

def display_tab1(df):
    # ... (existing display_tab1 code)
    st.header("📊 Data Overview & Initial Cleaning")
    if df is None:
        st.info("No dataset available for analysis. Please upload a CSV file.")
        return
    with st.expander("📄 Dataset Preview", expanded=True):
        st.subheader("First N Rows of Your Dataset")
        num_rows_to_display = st.slider("Select number of rows to preview", 5, min(50, df.shape[0]), 10, key="preview_rows_slider")
        st.dataframe(df.head(num_rows_to_display), use_container_width=True)
        st.info(f"Your dataset contains **{df.shape[0]} rows** and **{df.shape[1]} columns**.")
    with st.expander("📊 Basic Information & Quality Checks", expanded=True):
        st.subheader("Dataset Structure and Quality Summary")
        col_shape, col_duplicates, col_empty_cols = st.columns([1, 1, 1])
        with col_shape:
            st.markdown(f"- **Rows:** {df.shape[0]}")
            st.markdown(f"- **Columns:** {df.shape[1]}")
        with col_duplicates:
            duplicate_rows_count = df.duplicated().sum()
            st.markdown(f"- **Duplicate Rows:** {duplicate_rows_count}")
            if duplicate_rows_count > 0:
                if st.button("🧹 Drop Duplicates", key="drop_duplicates_btn"):
                    initial_rows = st.session_state.df.shape[0]
                    st.session_state.df.drop_duplicates(inplace=True)
                    st.session_state.df.reset_index(drop=True, inplace=True)
                    st.success(f"{initial_rows - st.session_state.df.shape[0]} duplicate rows removed.")
                    st.rerun()
            else:
                st.success("No duplicate rows found.")
        with col_empty_cols:
            empty_cols = df.columns[df.isnull().all()].tolist()
            if empty_cols:
                st.warning(f"⚠️ Columns with ALL missing values: `{', '.join(empty_cols)}`")
                if st.button("🗑️ Drop Empty Columns", key="drop_empty_cols_btn"):
                    initial_cols = st.session_state.df.shape[1]
                    st.session_state.df.dropna(axis=1, how='all', inplace=True)
                    st.success(f"Dropped {initial_cols - st.session_state.df.shape[1]} empty columns.")
                    st.rerun()
            else:
                st.info("No columns with all missing values found.")
        st.markdown("---")
        st.markdown("##### Column Information Summary")
        info_summary = pd.DataFrame({
            'Column': df.columns,
            'Non-Null Count': df.notnull().sum().values,
            'Data Type': df.dtypes.values,
            'Unique Values': df.nunique().values,
            'Missing Values': df.isnull().sum().values,
            'Missing %': (df.isnull().sum().values / df.shape[0] * 100).round(2)
        })
        st.dataframe(info_summary, use_container_width=True)
        st.markdown("---")
        st.markdown("##### Numerical Features Statistics (`.describe()`)")
        numeric_cols = df.select_dtypes(include=np.number).columns
        if not numeric_cols.empty:
            st.dataframe(df[numeric_cols].describe().T, use_container_width=True)
        else:
            st.info("No numerical columns found in the dataset.")
        st.markdown("---")
        st.markdown("##### Categorical Features Value Counts")
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if not categorical_cols.empty:
            selected_cat_col = st.selectbox(
                "Select a categorical column to view value counts:",
                categorical_cols,
                key="cat_val_counts_select"
            )
            if selected_cat_col:
                st.dataframe(df[selected_cat_col].value_counts().to_frame("Count"), use_container_width=True)
        else:
            st.info("No categorical columns found in the dataset.")


def display_tab2(df):
    # ... (existing display_tab2 code)
    st.header("📊 Univariate Analysis")
    st.markdown("Explore individual column distributions and characteristics.")
    if df is None:
        st.info("No dataset available. Please upload a CSV file.")
        return
    col_options = df.columns.tolist()
    selected_column = st.selectbox("Select a column for univariate analysis:", col_options, key="uni_col_select")
    if selected_column:
        st.subheader(f"Analysis for: {selected_column}")
        st.write(f"**Data Type:** {df[selected_column].dtype}")
        st.write(f"**Number of Unique Values:** {df[selected_column].nunique()}")
        st.write(f"**Missing Values:** {df[selected_column].isnull().sum()} ({df[selected_column].isnull().sum() / df.shape[0] * 100:.2f}%)")
        if pd.api.types.is_numeric_dtype(df[selected_column]):
            st.markdown("---")
            st.subheader("Numerical Distribution")
            st.write(df[selected_column].describe())
            fig = px.histogram(df, x=selected_column, title=f"Distribution of {selected_column}")
            st.plotly_chart(fig, use_container_width=True)
            fig = px.box(df, y=selected_column, title=f"Box Plot of {selected_column}")
            st.plotly_chart(fig, use_container_width=True)
        elif pd.api.types.is_object_dtype(df[selected_column]) or pd.api.types.is_categorical_dtype(df[selected_column]):
            st.markdown("---")
            st.subheader("Categorical Distribution")
            value_counts = df[selected_column].value_counts().reset_index()
            value_counts.columns = [selected_column, 'Count']
            st.dataframe(value_counts, use_container_width=True)
            fig = px.bar(value_counts, x=selected_column, y='Count', title=f"Value Counts of {selected_column}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("This column type is not supported for detailed univariate analysis yet.")


def display_tab3(df):
    # ... (existing display_tab3 code)
    st.header("📈 Bivariate Analysis")
    st.markdown("Examine relationships between two variables.")
    if df is None:
        st.info("No dataset available. Please upload a CSV file.")
        return
    all_columns = df.columns.tolist()
    col_x, col_y = st.columns(2)
    with col_x:
        x_axis = st.selectbox("Select X-axis column:", all_columns, key="bivar_x_select")
    with col_y:
        y_axis = st.selectbox("Select Y-axis column:", all_columns, key="bivar_y_select")
    if x_axis and y_axis:
        if x_axis == y_axis:
            st.warning("X-axis and Y-axis cannot be the same. Please select different columns.")
            return
        x_dtype = df[x_axis].dtype
        y_dtype = df[y_axis].dtype
        plot_type = None
        if pd.api.types.is_numeric_dtype(x_dtype) and pd.api.types.is_numeric_dtype(y_dtype):
            plot_type = "Scatter Plot"
            fig = px.scatter(df, x=x_axis, y=y_axis, title=f"Scatter Plot: {x_axis} vs {y_axis}")
            st.plotly_chart(fig, use_container_width=True)
            correlation = df[x_axis].corr(df[y_axis])
            st.info(f"**Correlation between {x_axis} and {y_axis}:** {correlation:.2f}")
        elif (pd.api.types.is_object_dtype(x_dtype) or pd.api.types.is_categorical_dtype(x_dtype)) and pd.api.types.is_numeric_dtype(y_dtype):
            plot_type = "Box Plot / Violin Plot"
            st.plotly_chart(px.box(df, x=x_axis, y=y_axis, title=f"Box Plot of {y_axis} by {x_axis}"), use_container_width=True)
            st.plotly_chart(px.violin(df, x=x_axis, y=y_axis, title=f"Violin Plot of {y_axis} by {x_axis}"), use_container_width=True)
        elif pd.api.types.is_numeric_dtype(x_dtype) and (pd.api.types.is_object_dtype(y_dtype) or pd.api.types.is_categorical_dtype(y_dtype)):
            plot_type = "Box Plot / Violin Plot (reversed)"
            st.plotly_chart(px.box(df, x=x_axis, y=y_axis, orientation='h', title=f"Box Plot of {x_axis} by {y_axis}"), use_container_width=True)
            st.plotly_chart(px.violin(df, x=x_axis, y=y_axis, orientation='h', title=f"Violin Plot of {x_axis} by {y_axis}"), use_container_width=True)
        elif (pd.api.types.is_object_dtype(x_dtype) or pd.api.types.is_categorical_dtype(x_dtype)) and \
             (pd.api.types.is_object_dtype(y_dtype) or pd.api.types.is_categorical_dtype(y_dtype)):
            plot_type = "Count Plot / Heatmap (Categorical)"
            st.subheader("Count Plot")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(data=df, x=x_axis, hue=y_axis, ax=ax)
            ax.set_title(f"Count Plot of {x_axis} by {y_axis}")
            ax.set_xlabel(x_axis)
            ax.set_ylabel("Count")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.subheader("Cross-tabulation (Counts)")
            cross_tab = pd.crosstab(df[x_axis], df[y_axis])
            st.dataframe(cross_tab, use_container_width=True)
            st.subheader("Heatmap of Cross-tabulation")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cross_tab, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
            ax.set_title(f"Heatmap of {x_axis} vs {y_axis} Counts")
            st.pyplot(fig)
            plt.close(fig)
        if plot_type:
            st.info(f"Recommended Plot Type: **{plot_type}**")
        else:
            st.warning("No standard plot type available for this combination of column types.")


def display_tab4(df):
    # ... (existing display_tab4 code)
    st.header("📈 Advanced Visualizations")
    st.markdown("Generate more complex plots to uncover deeper insights.")
    if df is None:
        st.info("No dataset available. Please upload a CSV file.")
        return
    chart_type = st.selectbox(
        "Select Chart Type:",
        ["Correlation Heatmap", "Pair Plot", "Scatter Matrix", "Box Plot"],
        key="adv_chart_select"
    )
    if chart_type == "Correlation Heatmap":
        st.subheader("Correlation Heatmap")
        numeric_df = df.select_dtypes(include=np.number)
        if not numeric_df.empty:
            corr = numeric_df.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5, ax=ax)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("No numerical columns available for correlation heatmap.")
    elif chart_type == "Pair Plot":
        st.subheader("Pair Plot")
        numeric_df = df.select_dtypes(include=np.number)
        if not numeric_df.empty:
            selected_pair_cols = st.multiselect(
                "Select numerical columns for Pair Plot (max 5 recommended):",
                numeric_df.columns.tolist(),
                key="pair_plot_cols"
            )
            if len(selected_pair_cols) > 1:
                if len(selected_pair_cols) > 5:
                    st.warning("Selecting too many columns for Pair Plot can be slow and hard to interpret.")
                fig = sns.pairplot(df[selected_pair_cols])
                st.pyplot(fig)
                plt.close(fig.figure) # FIX: Access the underlying figure
            else:
                st.info("Please select at least two numerical columns for a Pair Plot.")
        else:
            st.warning("No numerical columns available for Pair Plot.")
    elif chart_type == "Scatter Matrix":
        st.subheader("Scatter Matrix (Plotly)")
        numeric_df = df.select_dtypes(include=np.number)
        if not numeric_df.empty:
            selected_scatter_cols = st.multiselect(
                "Select numerical columns for Scatter Matrix (max 5 recommended):",
                numeric_df.columns.tolist(),
                key="scatter_matrix_cols"
            )
            if len(selected_scatter_cols) > 1:
                if len(selected_scatter_cols) > 5:
                    st.warning("Selecting too many columns for Scatter Matrix can be slow.")
                fig = px.scatter_matrix(df, dimensions=selected_scatter_cols, title="Scatter Matrix")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Please select at least two numerical columns for a Scatter Matrix.")
        else:
            st.warning("No numerical columns available for Scatter Matrix.")
    elif chart_type == "Box Plot":
        st.subheader("Box Plot")
        x_box = st.selectbox("X-axis (Categorical/None)", ['None'] + df.select_dtypes(include='object').columns.tolist(), key="x_box")
        y_box = st.selectbox("Y-axis (Numerical)", df.select_dtypes(include=np.number).columns.tolist(), key="y_box")
        if y_box:
            if x_box != 'None':
                fig = px.box(df, x=x_box, y=y_box, title=f"Box Plot of {y_box} by {x_box}")
            else:
                fig = px.box(df, y=y_box, title=f"Box Plot of {y_box}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please select a numerical Y-axis for the box plot.")


def display_tab5(df):
    """
    Displays the fifth tab of the Data Analyzer with options to download the processed dataset.
    """
    st.header("📥 Download Data")
    st.markdown("Download the currently analyzed and potentially cleaned dataset.")

    # Convert DataFrame to CSV
    # Ensure the dataframe used for download reflects any cleaning/imputation done
    current_df_for_download = st.session_state.get('df', df) # Use session state if updated, else original df
    csv_export = current_df_for_download.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Download Processed CSV",
        data=csv_export,
        file_name="processed_dataset.csv",
        mime="text/csv"
    )
    st.info("The downloaded CSV will reflect all transformations applied in the application.")


def reset_data_state():
    """Resets the DataFrame and all preprocessing states to their original uploaded state."""
    if 'original_df' in st.session_state and st.session_state.original_df is not None:
        st.session_state.df = st.session_state.original_df.copy()
        st.session_state.scaled_columns = []
        st.session_state.encoded_columns = []
        st.session_state.selected_features_for_model = []
        st.session_state.trained_model = None
        st.session_state.X_test = None
        st.session_state.y_test = None
        st.session_state.problem_type = None
        st.session_state.X_columns = None
        st.success("Data reset to original uploaded state.")
        st.rerun()
    else:
        st.warning("No original dataset found to reset to. Please upload a dataset first.")


def handle_missing_values(df):
    """
    Displays UI for missing value handling and applies selected strategy.
    Args:
        df (pd.DataFrame): The current DataFrame from session state.
    Returns:
        None (updates st.session_state.df directly)
    """
    st.subheader("Missing Value Handling")
    missing_values_df = df.isnull().sum().to_frame("Missing Count")
    missing_values_df = missing_values_df[missing_values_df["Missing Count"] > 0]

    if not missing_values_df.empty:
        st.markdown("Columns with missing values:")
        st.dataframe(missing_values_df, use_container_width=True)
        missing_cols = missing_values_df.index.tolist()

        impute_option = st.selectbox(
            "Choose Missing Value Strategy",
            ["None", "Drop Rows with Missing Values", "Impute with Mean (Numeric)", "Impute with Median (Numeric)", "Impute with Mode (All types)"],
            key="missing_strategy"
        )

        if impute_option != "None":
            if st.button("Apply Missing Value Strategy", key="apply_missing_btn"):
                with st.spinner("Applying strategy..."):
                    if impute_option == "Drop Rows with Missing Values":
                        initial_rows = st.session_state.df.shape[0]
                        st.session_state.df.dropna(inplace=True)
                        st.session_state.df.reset_index(drop=True, inplace=True)
                        st.success(f"Dropped {initial_rows - st.session_state.df.shape[0]} rows with missing values.")
                    elif "Impute" in impute_option:
                        for col in missing_cols:
                            if pd.api.types.is_numeric_dtype(st.session_state.df[col]):
                                if impute_option == "Impute with Mean (Numeric)":
                                    st.session_state.df[col].fillna(st.session_state.df[col].mean(), inplace=True)
                                    st.success(f"Imputed missing values in '{col}' with mean.")
                                elif impute_option == "Impute with Median (Numeric)":
                                    st.session_state.df[col].fillna(st.session_state.df[col].median(), inplace=True)
                                    st.success(f"Imputed missing values in '{col}' with median.")
                            elif impute_option == "Impute with Mode (All types)":
                                # Apply mode to non-numeric too
                                st.session_state.df[col].fillna(st.session_state.df[col].mode()[0], inplace=True)
                                st.success(f"Imputed missing values in '{col}' with mode.")
                            else:
                                st.warning(f"Skipping imputation for non-numeric column '{col}' with numeric-only strategy.")
                    st.session_state.trained_model = None # Invalidate model if data changes
                    st.rerun()
        else:
            st.info("Select a strategy and click 'Apply' to handle missing values.")
    else:
        st.success("✅ No missing values detected in the dataset.")


def handle_categorical_encoding(df):
    """
    Displays UI for categorical encoding and applies selected method.
    Args:
        df (pd.DataFrame): The current DataFrame from session state.
    Returns:
        None (updates st.session_state.df directly)
    """
    st.subheader("Categorical Feature Encoding")
    if 'encoded_columns' not in st.session_state:
        st.session_state.encoded_columns = []

    all_categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    available_cols_for_encoding = [col for col in all_categorical_cols if col not in st.session_state.encoded_columns]

    if available_cols_for_encoding:
        st.write("Available categorical columns for encoding:", ", ".join(available_cols_for_encoding))
        encoding_method = st.selectbox(
            "Choose Encoding Method",
            ["None", "Label Encoding (Ordinal/Binary)", "One-Hot Encoding (Nominal)"],
            key="encoding_method"
        )
        cols_to_encode = st.multiselect("Select columns to encode", available_cols_for_encoding, key="cols_to_encode")

        if encoding_method != "None" and cols_to_encode:
            if st.button("Apply Encoding", key="apply_encoding_btn"):
                with st.spinner("Applying encoding..."):
                    current_df_state = st.session_state.df.copy()

                    for col in cols_to_encode:
                        if encoding_method == "Label Encoding (Ordinal/Binary)":
                            le = LabelEncoder()
                            current_df_state[col] = le.fit_transform(current_df_state[col].astype(str))
                            st.success(f"Applied Label Encoding to '{col}'.")
                            st.session_state.encoded_columns.append(col)
                        elif encoding_method == "One-Hot Encoding (Nominal)":
                            try:
                                current_df_state = pd.get_dummies(current_df_state, columns=[col], prefix=col, prefix_sep='_', drop_first=True)
                                st.success(f"Applied One-Hot Encoding to '{col}'.")
                                st.session_state.encoded_columns.append(col)
                            except Exception as e:
                                st.error(f"Error applying One-Hot Encoding to '{col}': {e}")

                    st.session_state.df = current_df_state
                    st.session_state.trained_model = None
                    st.rerun()
        else:
            st.info("Select an encoding method and columns to apply.")
    else:
        st.success("All categorical columns have been encoded or no categorical columns detected for encoding.")

    if st.session_state.encoded_columns:
        st.markdown(f"**Currently Encoded Original Columns:** {', '.join(st.session_state.encoded_columns)}")


def handle_feature_scaling(df):
    """
    Displays UI for feature scaling and applies selected method.
    Args:
        df (pd.DataFrame): The current DataFrame from session state.
    Returns:
        None (updates st.session_state.df directly)
    """
    st.subheader("Feature Scaling")
    if 'scaled_columns' not in st.session_state:
        st.session_state.scaled_columns = []

    all_numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    available_cols_for_scaling = [col for col in all_numeric_cols if col not in st.session_state.scaled_columns]

    if available_cols_for_scaling:
        st.write("Available numeric columns for scaling:", ", ".join(available_cols_for_scaling))
        scaling_method = st.selectbox(
            "Choose Scaling Method",
            ["None", "StandardScaler", "MinMaxScaler"],
            key="scaling_method"
        )
        cols_to_scale_selection = st.multiselect("Select columns to scale", available_cols_for_scaling, key="cols_to_scale")

        if scaling_method != "None" and cols_to_scale_selection:
            if st.button("Apply Scaling", key="apply_scaling_btn"):
                with st.spinner("Applying scaling..."):
                    current_df_state = st.session_state.df.copy()

                    for col in cols_to_scale_selection:
                        scaler = None
                        if scaling_method == "StandardScaler":
                            scaler = StandardScaler()
                        elif scaling_method == "MinMaxScaler":
                            scaler = MinMaxScaler()

                        if scaler:
                            current_df_state[col] = scaler.fit_transform(current_df_state[[col]])
                            st.success(f"Applied {scaling_method} to '{col}'.")
                            st.session_state.scaled_columns.append(col)
                        else:
                            st.error("Invalid scaler selected.")

                    st.session_state.df = current_df_state
                    st.session_state.trained_model = None
                    st.rerun()
        else:
            st.info("Select a scaling method and columns to apply.")
    else:
        st.success("All numeric columns have been scaled or no numeric columns detected for scaling.")

    if st.session_state.scaled_columns:
        st.markdown(f"**Currently Scaled Columns:** {', '.join(st.session_state.scaled_columns)}")


def display_preprocessing_section(df):
    """
    Integrates all preprocessing functionalities into a single section.
    Args:
        df (pd.DataFrame): The current DataFrame from session state.
    """
    st.header("2. Data Preprocessing")

    if st.button("🔄 Reset Data to Original Uploaded State"):
        reset_data_state()

    preprocessing_tabs = st.tabs(["Missing Values", "Categorical Encoding", "Feature Scaling", "📥 Download Processed Data"]) # Added Download tab

    with preprocessing_tabs[0]:
        handle_missing_values(df)

    with preprocessing_tabs[1]:
        handle_categorical_encoding(df)

    with preprocessing_tabs[2]:
        handle_feature_scaling(df)

    with preprocessing_tabs[3]: # New tab for download
        display_tab5(df)

    st.markdown("---")


# --- New Functions for Model Building & Evaluation Section ---

def determine_problem_type(df, target_column):
    """
    Determines if the problem is classification or regression based on the target column.
    Stores the problem type in st.session_state.
    """
    if target_column is None:
        st.session_state.problem_type = None
        return

    target_series = df[target_column]
    unique_values = target_series.nunique()
    dtype = target_series.dtype

    # Heuristic for classification vs. regression
    if pd.api.types.is_numeric_dtype(dtype):
        # If numeric, check number of unique values and if they are discrete-like
        # A common heuristic is if unique values are few and integer-like, it's classification
        # For example, if unique values are 0, 1, 2, 3...
        if unique_values <= 20 and all(target_series.dropna().apply(lambda x: x == int(x))):
            st.session_state.problem_type = "Classification"
            st.info(f"Target column '{target_column}' is numeric with {unique_values} unique, integer-like values. "
                    f"Assumed **Classification** problem.")
        else:
            st.session_state.problem_type = "Regression"
            st.info(f"Target column '{target_column}' is numeric with {unique_values} unique values. "
                    f"Assumed **Regression** problem.")
    elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):
        st.session_state.problem_type = "Classification"
        st.info(f"Target column '{target_column}' is categorical with {unique_values} unique values. "
                f"Assumed **Classification** problem.")
    else:
        st.session_state.problem_type = None
        st.warning(f"Could not determine problem type for column '{target_column}' with data type {dtype}. "
                   "Please ensure it's numeric or categorical.")

    # Clear trained model if target/problem type changes
    if 'problem_type' in st.session_state and st.session_state.problem_type != st.session_state.get('prev_problem_type'):
        st.session_state.trained_model = None
    st.session_state.prev_problem_type = st.session_state.problem_type


def select_target_and_features(df):
    """
    Allows user to select target and features, with validation.
    Updates st.session_state.target_column, st.session_state.selected_features,
    st.session_state.problem_type.
    """
    st.subheader("Select Target and Features")

    all_columns = df.columns.tolist()

    # Target Variable Selection
    target_column = st.selectbox(
        "Select your Target Variable (Y):",
        ['None'] + all_columns,
        key="target_select",
        on_change=lambda: determine_problem_type(df, st.session_state.target_select) # Update problem type on change
    )

    if target_column == 'None':
        st.session_state.target_column = None
        st.session_state.problem_type = None
        st.session_state.selected_features = []
        st.warning("Please select a target variable to proceed.")
        return False # Indicate that selection is not complete/valid

    st.session_state.target_column = target_column
    determine_problem_type(df, target_column) # Ensure problem type is determined/updated

    # Feature Selection
    available_features = [col for col in all_columns if col != target_column]
    selected_features = st.multiselect(
        "Select Feature Variables (X):",
        available_features,
        default=st.session_state.get('selected_features', []), # Keep previous selection
        key="features_select"
    )
    st.session_state.selected_features = selected_features

    # Restrictions/Validations
    if not selected_features:
        st.warning("Please select at least one feature variable.")
        return False

    # Check for non-numeric/non-encoded features
    non_numeric_features = []
    for feature in selected_features:
        # Check if feature is not numeric AND not in the list of columns that were one-hot encoded
        # Label encoded columns become numeric, so they are covered by is_numeric_dtype
        if not pd.api.types.is_numeric_dtype(df[feature].dtype) and feature not in st.session_state.encoded_columns:
            non_numeric_features.append(feature)

    if non_numeric_features:
        st.error(f"⚠️ Selected features {', '.join(non_numeric_features)} are not numeric "
                 f"and have not been encoded. Please encode them in the 'Data Preprocessing' tab "
                 f"or select only numeric/encoded features.")
        return False

    # Check for missing values in selected features or target
    # Only check if there are actual missing values in the relevant columns
    data_for_check = df[selected_features + [target_column]]
    missing_in_selection = data_for_check.isnull().sum()
    missing_in_selection = missing_in_selection[missing_in_selection > 0]
    if not missing_in_selection.empty:
        st.error(f"⚠️ Missing values detected in selected target or features: "
                 f"{', '.join(missing_in_selection.index.tolist())}. "
                 f"Please handle them in the 'Data Preprocessing' tab.")
        return False

    return True # All checks passed


def display_model_selection_and_params():
    """
    Displays model selection dropdown and dynamic parameter inputs based on problem type.
    Updates st.session_state.selected_model_name and st.session_state.selected_model_params.
    """
    st.subheader("Choose Model and Tune Parameters")

    problem_type = st.session_state.get('problem_type')
    if problem_type is None:
        st.info("Please select a target variable first to determine the problem type.")
        st.session_state.selected_model_name = None
        st.session_state.selected_model_params = {}
        return

    model_options = []
    if problem_type == "Regression":
        model_options = [
            "Linear Regression", "Ridge Regression", "Lasso Regression",
            "Random Forest Regressor", "Gradient Boosting Regressor",
            "KNN Regressor", "SVM Regressor"
        ]
        if XGBRegressor is not None:
            model_options.append("XGBoost Regressor")
    elif problem_type == "Classification":
        model_options = [
            "Logistic Regression", "Random Forest Classifier", "Gradient Boosting Classifier",
            "KNN Classifier", "SVM Classifier"
        ]
        if XGBClassifier is not None:
            model_options.append("XGBoost Classifier")

    selected_model_name = st.selectbox(
        f"Select a {problem_type} Model:",
        ['None'] + model_options,
        key="model_select",
        on_change=lambda: st.session_state.update(selected_model_params={}, trained_model=None) # Clear params/model on new selection
    )

    st.session_state.selected_model_name = selected_model_name
    # Initialize selected_model_params for the chosen model if it's new
    if selected_model_name != 'None' and selected_model_name not in st.session_state.selected_model_params:
        st.session_state.selected_model_params = {} # Reset if a new model is chosen

    if selected_model_name == 'None':
        st.info("Please select a model.")
        return

    st.markdown("##### Model Parameters")
    # Dynamic parameter inputs based on selected model
    if selected_model_name == "Linear Regression":
        st.info("Linear Regression typically does not require hyperparameter tuning for basic usage.")
        st.session_state.selected_model_params['fit_intercept'] = st.checkbox(
            "Fit Intercept", value=st.session_state.selected_model_params.get('fit_intercept', True), key="lr_fit_intercept"
        )
        # No other common tunable parameters for simple Linear Regression via scikit-learn
    elif selected_model_name == "Logistic Regression":
        st.session_state.selected_model_params['C'] = st.slider(
            "C (Inverse of regularization strength):", 0.01, 10.0, float(st.session_state.selected_model_params.get('C', 1.0)), key="lr_C"
        )
        st.session_state.selected_model_params['solver'] = st.selectbox(
            "Solver:", ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'],
            index=['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'].index(st.session_state.selected_model_params.get('solver', 'liblinear')),
            key="lr_solver"
        )
        st.session_state.selected_model_params['penalty'] = st.selectbox(
            "Penalty:", ['l1', 'l2', 'elasticnet', 'none'],
            index=['l1', 'l2', 'elasticnet', 'none'].index(st.session_state.selected_model_params.get('penalty', 'l2')),
            key="lr_penalty"
        )
        st.session_state.selected_model_params['max_iter'] = st.slider(
            "Max Iterations:", 100, 1000, int(st.session_state.selected_model_params.get('max_iter', 100)), key="lr_max_iter"
        )
        if st.session_state.selected_model_params['penalty'] == 'elasticnet':
            st.session_state.selected_model_params['l1_ratio'] = st.slider(
                "L1 Ratio (ElasticNet only):", 0.0, 1.0, float(st.session_state.selected_model_params.get('l1_ratio', 0.5)), key="lr_l1_ratio"
            )
        # Add a warning for incompatible solver-penalty combinations
        if (st.session_state.selected_model_params['penalty'] == 'l1' and st.session_state.selected_model_params['solver'] not in ['liblinear', 'saga']) or \
           (st.session_state.selected_model_params['penalty'] == 'elasticnet' and st.session_state.selected_model_params['solver'] != 'saga') or \
           (st.session_state.selected_model_params['penalty'] == 'none' and st.session_state.selected_model_params['solver'] == 'liblinear'):
            st.warning(f"Selected penalty '{st.session_state.selected_model_params['penalty']}' and solver '{st.session_state.selected_model_params['solver']}' might be incompatible. Refer to scikit-learn docs.")

    elif selected_model_name == "Ridge Regression":
        st.session_state.selected_model_params['alpha'] = st.slider(
            "Alpha (Regularization strength):", 0.01, 10.0, float(st.session_state.selected_model_params.get('alpha', 1.0)), key="ridge_alpha"
        )
        st.session_state.selected_model_params['fit_intercept'] = st.checkbox(
            "Fit Intercept", value=st.session_state.selected_model_params.get('fit_intercept', True), key="ridge_fit_intercept"
        )
        st.session_state.selected_model_params['solver'] = st.selectbox(
            "Solver:", ['auto', 'svd', 'cholesky', 'lsqr', 'sag', 'saga'],
            index=['auto', 'svd', 'cholesky', 'lsqr', 'sag', 'saga'].index(st.session_state.selected_model_params.get('solver', 'auto')),
            key="ridge_solver"
        )
    elif selected_model_name == "Lasso Regression":
        st.session_state.selected_model_params['alpha'] = st.slider(
            "Alpha (Regularization strength):", 0.01, 10.0, float(st.session_state.selected_model_params.get('alpha', 1.0)), key="lasso_alpha"
        )
        st.session_state.selected_model_params['fit_intercept'] = st.checkbox(
            "Fit Intercept", value=st.session_state.selected_model_params.get('fit_intercept', True), key="lasso_fit_intercept"
        )
        st.session_state.selected_model_params['selection'] = st.selectbox(
            "Selection Method:", ['cyclic', 'random'],
            index=['cyclic', 'random'].index(st.session_state.selected_model_params.get('selection', 'cyclic')),
            key="lasso_selection"
        )
    elif "Random Forest" in selected_model_name:
        st.session_state.selected_model_params['n_estimators'] = st.slider(
            "Number of Estimators:", 10, 500, int(st.session_state.selected_model_params.get('n_estimators', 100)), key="rf_n_estimators"
        )
        st.session_state.selected_model_params['max_depth'] = st.slider(
            "Max Depth (None for unlimited):", 1, 50, int(st.session_state.selected_model_params.get('max_depth', 10)), key="rf_max_depth"
        )
        st.session_state.selected_model_params['min_samples_split'] = st.slider(
            "Min Samples Split:", 2, 20, int(st.session_state.selected_model_params.get('min_samples_split', 2)), key="rf_min_split"
        )
        st.session_state.selected_model_params['min_samples_leaf'] = st.slider(
            "Min Samples Leaf:", 1, 20, int(st.session_state.selected_model_params.get('min_samples_leaf', 1)), key="rf_min_leaf"
        )
        # FIX: Changed 'auto' to 'sqrt' as default and adjusted options
        st.session_state.selected_model_params['max_features'] = st.selectbox(
            "Max Features:", ['sqrt', 'log2', None, 0.5, 0.7, 0.9], # 'auto' removed, None added
            index=['sqrt', 'log2', None, 0.5, 0.7, 0.9].index(st.session_state.selected_model_params.get('max_features', 'sqrt')),
            key="rf_max_features"
        )
        st.session_state.selected_model_params['bootstrap'] = st.checkbox(
            "Bootstrap Samples", value=st.session_state.selected_model_params.get('bootstrap', True), key="rf_bootstrap"
        )
    elif "Gradient Boosting" in selected_model_name:
        st.session_state.selected_model_params['n_estimators'] = st.slider(
            "Number of Estimators:", 10, 500, int(st.session_state.selected_model_params.get('n_estimators', 100)), key="gb_n_estimators"
        )
        st.session_state.selected_model_params['learning_rate'] = st.slider(
            "Learning Rate:", 0.01, 0.5, float(st.session_state.selected_model_params.get('learning_rate', 0.1)), key="gb_lr"
        )
        st.session_state.selected_model_params['max_depth'] = st.slider(
            "Max Depth:", 1, 10, int(st.session_state.selected_model_params.get('max_depth', 3)), key="gb_max_depth"
        )
        st.session_state.selected_model_params['subsample'] = st.slider(
            "Subsample Ratio:", 0.5, 1.0, float(st.session_state.selected_model_params.get('subsample', 1.0)), key="gb_subsample"
        )
        if problem_type == "Regression":
            st.session_state.selected_model_params['loss'] = st.selectbox(
                "Loss Function:", ['ls', 'lad', 'huber', 'quantile'],
                index=['ls', 'lad', 'huber', 'quantile'].index(st.session_state.selected_model_params.get('loss', 'ls')),
                key="gb_loss_reg"
            )
        elif problem_type == "Classification":
            st.session_state.selected_model_params['loss'] = st.selectbox(
                "Loss Function:", ['deviance', 'exponential'],
                index=['deviance', 'exponential'].index(st.session_state.selected_model_params.get('loss', 'deviance')),
                key="gb_loss_clf"
            )
    elif "KNN" in selected_model_name:
        st.session_state.selected_model_params['n_neighbors'] = st.slider(
            "Number of Neighbors:", 1, 20, int(st.session_state.selected_model_params.get('n_neighbors', 5)), key="knn_neighbors"
        )
        st.session_state.selected_model_params['weights'] = st.selectbox(
            "Weights:", ['uniform', 'distance'],
            index=['uniform', 'distance'].index(st.session_state.selected_model_params.get('weights', 'uniform')),
            key="knn_weights"
        )
        st.session_state.selected_model_params['algorithm'] = st.selectbox(
            "Algorithm:", ['auto', 'ball_tree', 'kd_tree', 'brute'],
            index=['auto', 'ball_tree', 'kd_tree', 'brute'].index(st.session_state.selected_model_params.get('algorithm', 'auto')),
            key="knn_algorithm"
        )
        st.session_state.selected_model_params['p'] = st.slider(
            "P (Minkowski metric, 1=Manhattan, 2=Euclidean):", 1, 2, int(st.session_state.selected_model_params.get('p', 2)), key="knn_p"
        )
    elif "SVM" in selected_model_name:
        st.session_state.selected_model_params['C'] = st.slider(
            "C (Regularization parameter):", 0.1, 10.0, float(st.session_state.selected_model_params.get('C', 1.0)), key="svm_C"
        )
        st.session_state.selected_model_params['kernel'] = st.selectbox(
            "Kernel:", ['rbf', 'linear', 'poly', 'sigmoid'],
            index=['rbf', 'linear', 'poly', 'sigmoid'].index(st.session_state.selected_model_params.get('kernel', 'rbf')),
            key="svm_kernel"
        )
        st.session_state.selected_model_params['gamma'] = st.selectbox(
            "Gamma (Kernel coefficient):", ['scale', 'auto', 0.01, 0.1, 1.0],
            index=['scale', 'auto', 0.01, 0.1, 1.0].index(st.session_state.selected_model_params.get('gamma', 'scale')),
            key="svm_gamma"
        )
        if selected_model_name == "SVM Regressor": # SVR specific
             st.session_state.selected_model_params['epsilon'] = st.slider(
                "Epsilon (SVR specific):", 0.01, 1.0, float(st.session_state.selected_model_params.get('epsilon', 0.1)), key="svr_epsilon"
            )
        if selected_model_name == "SVM Classifier":
            st.session_state.selected_model_params['probability'] = st.checkbox(
                "Enable Probability Estimates (slows training)", value=st.session_state.selected_model_params.get('probability', False), key="svc_probability"
            )
    elif "XGBoost" in selected_model_name:
        if XGBRegressor is None: # Check if XGBoost was imported successfully
            st.error("XGBoost is not available. Please install it (`pip install xgboost`).")
            st.session_state.selected_model_params = {}
            return

        st.session_state.selected_model_params['n_estimators'] = st.slider(
            "Number of Estimators:", 10, 500, int(st.session_state.selected_model_params.get('n_estimators', 100)), key="xgb_n_estimators"
        )
        st.session_state.selected_model_params['learning_rate'] = st.slider(
            "Learning Rate:", 0.01, 0.5, float(st.session_state.selected_model_params.get('learning_rate', 0.1)), key="xgb_lr"
        )
        st.session_state.selected_model_params['max_depth'] = st.slider(
            "Max Depth:", 1, 10, int(st.session_state.selected_model_params.get('max_depth', 3)), key="xgb_max_depth"
        )
        st.session_state.selected_model_params['subsample'] = st.slider(
            "Subsample Ratio:", 0.5, 1.0, float(st.session_state.selected_model_params.get('subsample', 1.0)), key="xgb_subsample"
        )
        st.session_state.selected_model_params['colsample_bytree'] = st.slider(
            "Colsample By Tree Ratio:", 0.5, 1.0, float(st.session_state.selected_model_params.get('colsample_bytree', 1.0)), key="xgb_colsample"
        )
        st.session_state.selected_model_params['gamma'] = st.slider(
            "Gamma (Min loss reduction to make a further partition):", 0.0, 1.0, float(st.session_state.selected_model_params.get('gamma', 0.0)), key="xgb_gamma"
        )
        st.session_state.selected_model_params['reg_lambda'] = st.slider(
            "Lambda (L2 regularization):", 0.0, 10.0, float(st.session_state.selected_model_params.get('reg_lambda', 1.0)), key="xgb_lambda"
        )
        st.session_state.selected_model_params['reg_alpha'] = st.slider(
            "Alpha (L1 regularization):", 0.0, 10.0, float(st.session_state.selected_model_params.get('reg_alpha', 0.0)), key="xgb_alpha"
        )


def get_model_instance(model_name, params):
    """Returns an instance of the selected model with the given parameters."""
    model = None
    if model_name == "Linear Regression":
        model = LinearRegression(**params)
    elif model_name == "Ridge Regression":
        model = Ridge(**params)
    elif model_name == "Lasso Regression":
        model = Lasso(**params)
    elif model_name == "Random Forest Regressor":
        model = RandomForestRegressor(random_state=42, **params)
    elif model_name == "Gradient Boosting Regressor":
        model = GradientBoostingRegressor(random_state=42, **params)
    elif model_name == "KNN Regressor":
        model = KNeighborsRegressor(**params)
    elif model_name == "SVM Regressor":
        model = SVR(**params)
    elif model_name == "XGBoost Regressor":
        if XGBRegressor:
            model = XGBRegressor(random_state=42, **params)
        else:
            st.error("XGBoost Regressor is not available.")
            return None
    elif model_name == "Logistic Regression":
        model = LogisticRegression(random_state=42, **params)
    elif model_name == "Random Forest Classifier":
        model = RandomForestClassifier(random_state=42, **params)
    elif model_name == "Gradient Boosting Classifier":
        model = GradientBoostingClassifier(random_state=42, **params)
    elif model_name == "KNN Classifier":
        model = KNeighborsClassifier(**params)
    elif model_name == "SVM Classifier":
        model = SVC(random_state=42, **params)
    elif model_name == "XGBoost Classifier":
        if XGBClassifier:
            # For multi-class classification, `use_label_encoder=False` and `eval_metric='logloss'` are common
            # to suppress future warnings and set a default metric.
            model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', **params)
        else:
            st.error("XGBoost Classifier is not available.")
            return None
    return model


def train_and_evaluate_model(X, y, test_size, problem_type, model_name, model_params):
    """
    Performs train-test split, trains the selected model, and evaluates it.
    Displays metrics and visualizations.
    Updates st.session_state.trained_model, X_test, y_test.
    """
    st.subheader("Model Training & Evaluation Results")

    if X.empty or y.empty:
        st.error("Cannot train model: Features (X) or Target (y) data is empty.")
        return

    try:
        # Train-Test Split
        stratify_y = y if problem_type == "Classification" else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify_y
        )
        st.session_state.X_test = X_test # Store for potential future use (e.g., prediction)
        st.session_state.y_test = y_test

        st.info(f"Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples).")

        # Get model instance
        model = get_model_instance(model_name, model_params)
        if model is None:
            st.error("Failed to create model instance. Please check model selection or XGBoost installation.")
            return

        with st.spinner(f"Training {model_name}..."):
            model.fit(X_train, y_train)
            st.session_state.trained_model = model # Store the trained model
            st.session_state.X_columns = X.columns.tolist() # Store feature names for prediction

        st.success(f"Model '{model_name}' trained successfully!")

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate and Visualize
        if problem_type == "Regression":
            st.markdown("##### Regression Metrics")
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            metrics_data = {
                "Metric": ["MAE", "MSE", "RMSE", "R-squared"],
                "Value": [f"{mae:.3f}", f"{mse:.3f}", f"{rmse:.3f}", f"{r2:.3f}"]
            }
            st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)

            st.markdown("##### Actual vs. Predicted Plot")
            fig = px.scatter(x=y_test, y=y_pred,
                             labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                             title=f'Actual vs. Predicted Values for {model_name}')
            fig.add_shape(type="line", x0=y_test.min(), y0=y_test.min(),
                          x1=y_test.max(), y1=y_test.max(),
                          line=dict(color="red", width=2, dash="dash"),
                          name="Ideal Fit")
            st.plotly_chart(fig, use_container_width=True)

        elif problem_type == "Classification":
            st.markdown("##### Classification Metrics")
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            metrics_data = {
                "Metric": ["Accuracy", "Precision (Weighted)", "Recall (Weighted)", "F1-Score (Weighted)"],
                "Value": [f"{accuracy:.3f}", f"{precision:.3f}", f"{recall:.3f}", f"{f1:.3f}"]
            }
            st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)

            st.markdown("##### Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                        xticklabels=model.classes_, yticklabels=model.classes_)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title(f'Confusion Matrix for {model_name}')
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("##### Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred during model training or evaluation: {e}")
        st.exception(e) # Display full traceback for debugging


def display_model_upload_and_details():
    """
    Allows user to upload a pre-trained model and displays its details.
    """
    st.subheader("Upload and Inspect Pre-trained Model")
    st.markdown("Upload a `.joblib` file to inspect its type and parameters.")

    uploaded_model_file = st.file_uploader(
        "Upload Model (.joblib)",
        type=["joblib"],
        key="model_uploader"
    )

    if uploaded_model_file is not None:
        try:
            with st.spinner("Loading model..."):
                # Load the model from the uploaded file
                loaded_model = joblib.load(uploaded_model_file)
            
            st.success("Model loaded successfully!")
            st.markdown("---")
            st.subheader("Model Details")

            # Display basic model type
            st.write(f"**Model Type:** `{type(loaded_model).__name__}`")

            # Display model parameters if available
            if hasattr(loaded_model, 'get_params'):
                st.markdown("##### Model Parameters:")
                params = loaded_model.get_params()
                params_df = pd.DataFrame(params.items(), columns=['Parameter', 'Value'])
                st.dataframe(params_df, use_container_width=True)
            else:
                st.info("This model does not have a 'get_params()' method to display parameters.")
            
            # Optionally display feature importances or coefficients if applicable
            if hasattr(loaded_model, 'feature_importances_') and st.session_state.get('X_columns'):
                st.markdown("##### Feature Importances:")
                feature_importances = pd.Series(loaded_model.feature_importances_, index=st.session_state.X_columns)
                st.dataframe(feature_importances.sort_values(ascending=False).to_frame("Importance"), use_container_width=True)
            elif hasattr(loaded_model, 'coef_') and st.session_state.get('X_columns'):
                st.markdown("##### Coefficients:")
                # Handle multi-class coefficients if necessary
                if loaded_model.coef_.ndim > 1:
                    st.warning("Multi-class coefficients detected. Displaying first class coefficients.")
                    coefficients = pd.Series(loaded_model.coef_[0], index=st.session_state.X_columns)
                else:
                    coefficients = pd.Series(loaded_model.coef_, index=st.session_state.X_columns)
                st.dataframe(coefficients.sort_values(ascending=False).to_frame("Coefficient"), use_container_width=True)

        except Exception as e:
            st.error(f"Error loading or inspecting model: {e}")
            st.warning("Please ensure the uploaded file is a valid, pre-trained scikit-learn model saved with `joblib`.")
            st.exception(e) # Display full traceback for debugging


def display_model_building_section(df):
    """
    Main function to display the Model Building & Evaluation section.
    Orchestrates target/feature selection, model choice, training, and evaluation.
    """
    st.header("3. Model Building & Evaluation")

    # The main tabs for Model Building & Evaluation
    model_building_tabs = st.tabs([
        "🚀 Train & Evaluate Model",
        "💾 Upload & Inspect Model" # New tab for model upload
    ])

    with model_building_tabs[0]:
        # This section now has its own check for dataframe presence
        if df is None:
            st.info("Please upload a dataset in the 'Data Overview' tab to train or evaluate a model.")
            return # Exit this tab's rendering if no df

        # 1. Target and Feature Selection
        features_and_target_selected = select_target_and_features(df)

        if not features_and_target_selected:
            return # Stop if target/features are not valid or not selected

        # Ensure problem_type is determined and stored
        problem_type = st.session_state.get('problem_type')
        if problem_type is None:
            st.warning("Problem type could not be determined. Please check your target variable.")
            return

        st.markdown("---")

        # 2. Model Selection and Parameter Tuning
        display_model_selection_and_params()

        selected_model_name = st.session_state.get('selected_model_name')
        selected_model_params = st.session_state.get('selected_model_params', {})

        if selected_model_name is None or selected_model_name == 'None':
            st.info("Please select a model to train.")
            return

        st.markdown("---")

        # 3. Train-Test Split Configuration
        st.subheader("Train-Test Split Configuration")
        test_size = st.slider("Test Data Percentage:", 0.1, 0.5, float(st.session_state.get('test_size', 0.2)), 0.05, key="test_size_slider")
        st.session_state.test_size = test_size

        st.markdown("---")

        # 4. Train Model Button
        if st.button(f"🚀 Train {selected_model_name} Model", key="train_model_btn"):
            # Prepare X and y
            X = df[st.session_state.selected_features]
            y = df[st.session_state.target_column]

            # Final checks before training
            if X.empty or y.empty:
                st.error("Features (X) or Target (y) data is empty after selection. Cannot train model.")
                return
            if X.isnull().any().any() or y.isnull().any():
                st.error("Missing values still detected in selected features or target. Please handle them in the 'Data Preprocessing' tab.")
                return
            if not pd.api.types.is_numeric_dtype(X.values) and not all(col in st.session_state.encoded_columns for col in st.session_state.selected_features):
                 st.error("Features must be numeric. Ensure all categorical features are encoded.")
                 return
            if not pd.api.types.is_numeric_dtype(y.dtype) and problem_type == "Regression":
                st.error("Regression target must be numeric.")
                return

            train_and_evaluate_model(X, y, test_size, problem_type, selected_model_name, selected_model_params)

        st.markdown("---")

        # 5. Download Model Button
        if st.session_state.get('trained_model') is not None:
            st.success(f"Model '{st.session_state.selected_model_name}' is currently trained and ready for potential predictions.")
            st.write("Trained Model Object:", st.session_state.trained_model)

            # Create a download button for the trained model
            try:
                # Serialize the model using joblib.dump to an in-memory bytes buffer
                buffer = io.BytesIO()
                joblib.dump(st.session_state.trained_model, buffer)
                model_bytes = buffer.getvalue()

                st.download_button(
                    label="💾 Download Trained Model",
                    data=model_bytes,
                    file_name=f"{st.session_state.selected_model_name.replace(' ', '_').lower()}_model.joblib",
                    mime="application/octet-stream",
                    key="download_model_btn"
                )
                st.info("Click the button above to download the trained model.")
            except Exception as e:
                st.error(f"Error preparing model for download: {e}")
                st.warning("Model serialization failed. Ensure the trained model is picklable.")
    
    with model_building_tabs[1]: # Content for the new "Upload & Inspect Model" tab
        display_model_upload_and_details()


# --- New Wrapper Function for Data Analysis Tabs ---
def display_all_data_tabs(df):
    """
    Groups all data analysis and visualization tabs into a single section.
    """
    st.header("1. Data Analysis & Visualization")

    data_tabs = st.tabs([
        "📊 Data Overview & Cleaning",
        "🔬 Univariate Analysis",
        "📈 Bivariate Analysis",
        "🎯 Advanced Visualizations"
        # Removed "📥 Download Data" from here
    ])

    with data_tabs[0]:
        display_tab1(df)
    with data_tabs[1]:
        display_tab2(df)
    with data_tabs[2]:
        display_tab3(df)
    with data_tabs[3]:
        display_tab4(df)

    st.markdown("---")
