import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

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

def display_tab1(df): # df here is already st.session_state.df from the caller
    """Displays the first tab of the Data Analyzer with data overview and initial cleaning options."""

    st.header("📊 Data Overview & Initial Cleaning")

    with st.expander("📄 Dataset Preview", expanded=True):
        st.subheader("First 20 Rows of Your Dataset")
        st.dataframe(df.head(20), use_container_width=True)
        st.info(f"Your dataset contains **{df.shape[0]} rows** and **{df.shape[1]} columns**.")

    with st.expander("📊 Basic Information & Quality Checks", expanded=True):
        st.subheader("Dataset Structure and Quality Summary")

        # Custom Info Table
        data_types_list = [str(dtype) for dtype in df.dtypes]

        info_summary = pd.DataFrame({
            'Column': df.columns,
            'Non-Null Count': df.notnull().sum().values,
            'Data Type': data_types_list,
            'Unique Values': df.nunique().values,
            'Missing Values': df.isnull().sum().values,
            'Missing %': (df.isnull().sum() / len(df) * 100).round(2).values
        })
        st.dataframe(info_summary, use_container_width=True)

        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
        col_kpi1.metric("Total Rows", f"{df.shape[0]:,}")
        col_kpi2.metric("Total Columns", f"{df.shape[1]:,}")
        col_kpi3.metric("Numeric Columns", f"{df.select_dtypes(include=np.number).shape[1]:,}")
        col_kpi4.metric("Categorical Columns", f"{df.select_dtypes(include='object').shape[1]:,}")

        # Duplicate Rows Handling
        st.subheader("🧹 Duplicate Rows")
        duplicate_rows_count = df.duplicated().sum()
        st.markdown(f"- **Number of Duplicate Rows:** `{duplicate_rows_count:,}`")

        if duplicate_rows_count > 0:
            st.warning(f"There are {duplicate_rows_count} duplicate rows in your dataset.")
            if st.button("Drop Duplicates Now"):
                df_modified = df.drop_duplicates().copy()
                st.session_state['df'] = df_modified
                st.success("✅ Duplicate rows removed successfully!")
                st.rerun()
        else:
            st.success("✅ No duplicate rows detected in your dataset.")

    with st.expander("🕳️ Missing Value Analysis & Imputation", expanded=True):
        st.subheader("Detailed Missing Value Overview")

        # --- Re-evaluate missing values after imputation (if applicable) ---
        # This function will be called on every rerun. The df passed here
        # is already the one from session_state, so it's the latest version.
        current_missing_values = df.isnull().sum()
        current_missing_values = current_missing_values[current_missing_values > 0].sort_values(ascending=False)

        if not current_missing_values.empty:
            st.warning("Missing values detected in the following columns:")
            missing_df = current_missing_values.to_frame("Missing Count")
            missing_df["Percentage"] = (missing_df["Missing Count"] / len(df) * 100).round(2)
            st.dataframe(missing_df, use_container_width=True)

            fig_missing = px.bar(missing_df, x=missing_df.index, y="Percentage",
                                 title="Percentage of Missing Values per Column",
                                 labels={'index': 'Column', 'Percentage': 'Missing Percentage (%)'})
            st.plotly_chart(fig_missing, use_container_width=True)

            st.subheader("Imputation Options")
            impute_checkbox = st.checkbox("Enable Missing Value Imputation")
            if impute_checkbox:
                # Identify columns with NaNs based on the *current* df for display
                numeric_cols_with_nan = df.select_dtypes(include=np.number).columns[df.select_dtypes(include=np.number).isnull().any()].tolist()
                categorical_cols_with_nan = df.select_dtypes(include='object').columns[df.select_dtypes(include='object').isnull().any()].tolist()

                if numeric_cols_with_nan:
                    numeric_impute_method = st.radio("Numeric Imputation Method:", ["Mean", "Median", "Drop"], key="num_impute_method")
                    if numeric_impute_method == "Mean":
                        st.info(f"Numeric columns {numeric_cols_with_nan} will be imputed with their mean.")
                    elif numeric_impute_method == "Median":
                        st.info(f"Numeric columns {numeric_cols_with_nan} will be imputed with their median.")
                    elif numeric_impute_method == "Drop":
                        st.info(f"Rows with missing values in numeric columns {numeric_cols_with_nan} will be dropped.")

                if categorical_cols_with_nan:
                    categorical_impute_method = st.radio("Categorical Imputation Method:", ["Mode", "Constant", "Drop"], key="cat_impute_method")
                    if categorical_impute_method == "Mode":
                        st.info(f"Categorical columns {categorical_cols_with_nan} will be imputed with their mode.")
                    elif categorical_impute_method == "Constant":
                        fill_value = st.text_input("Enter constant value for categorical imputation:", "Missing", key="st_text_input_constant")
                        st.info(f"Categorical columns {categorical_cols_with_nan} will be imputed with '{fill_value}'.")
                    elif categorical_impute_method == "Drop":
                        st.info(f"Rows with missing values in categorical columns {categorical_cols_with_nan} will be dropped.")

                if st.button("Apply Imputation"):
                    df_to_impute = df.copy()

                    numeric_cols_with_nan_apply = df_to_impute.select_dtypes(include=np.number).columns[df_to_impute.select_dtypes(include=np.number).isnull().any()].tolist()
                    categorical_cols_with_nan_apply = df_to_impute.select_dtypes(include='object').columns[df_to_impute.select_dtypes(include='object').isnull().any()].tolist()

                    imputation_applied = False # Flag to check if any imputation was done

                    if numeric_cols_with_nan_apply:
                        selected_numeric_method = st.session_state.get("num_impute_method", "Mean")
                        if selected_numeric_method == "Mean":
                            df_to_impute[numeric_cols_with_nan_apply] = df_to_impute[numeric_cols_with_nan_apply].fillna(df_to_impute[numeric_cols_with_nan_apply].mean())
                            imputation_applied = True
                        elif selected_numeric_method == "Median":
                            df_to_impute[numeric_cols_with_nan_apply] = df_to_impute[numeric_cols_with_nan_apply].fillna(df_to_impute[numeric_cols_with_nan_apply].median())
                            imputation_applied = True
                        elif selected_numeric_method == "Drop":
                            initial_rows = df_to_impute.shape[0]
                            df_to_impute.dropna(subset=numeric_cols_with_nan_apply, inplace=True)
                            if df_to_impute.shape[0] < initial_rows: # Check if rows were actually dropped
                                imputation_applied = True


                    if categorical_cols_with_nan_apply:
                        selected_categorical_method = st.session_state.get("cat_impute_method", "Mode")
                        if selected_categorical_method == "Mode":
                            for col in categorical_cols_with_nan_apply:
                                df_to_impute[col] = df_to_impute[col].fillna(df_to_impute[col].mode()[0])
                            imputation_applied = True
                        elif selected_categorical_method == "Constant":
                            fill_value = st.session_state.get("st_text_input_constant", "Missing")
                            for col in categorical_cols_with_nan_apply:
                                df_to_impute[col] = df_to_impute[col].fillna(fill_value)
                            imputation_applied = True
                        elif selected_categorical_method == "Drop":
                            initial_rows = df_to_impute.shape[0]
                            df_to_impute.dropna(subset=categorical_cols_with_nan_apply, inplace=True)
                            if df_to_impute.shape[0] < initial_rows: # Check if rows were actually dropped
                                imputation_applied = True

                    st.session_state['df'] = df_to_impute # Update session state with the fully imputed DataFrame

                    # Immediately re-check missing values *after* the update to session state
                    # This is the key to seeing immediate feedback
                    post_impute_missing_values = st.session_state['df'].isnull().sum()
                    post_impute_missing_values = post_impute_missing_values[post_impute_missing_values > 0]

                    if imputation_applied:
                        if post_impute_missing_values.empty:
                            st.success("✅ Missing values imputed/handled successfully! No missing values detected after processing.")
                        else:
                            st.warning("⚠️ Imputation applied, but some missing values may remain or new ones appeared. Please review the 'Detailed Missing Value Overview' above.")
                    else:
                        st.info("No imputation applied as no missing values were found for selected columns or no method was chosen.")

                    st.rerun() # Trigger a rerun to fully refresh the UI with the new data
        else:
            st.success("✅ No missing values detected in your dataset.")

def display_tab2(df):

    """Displays the second tab of the Data Analyzer with univariate analysis and visualizations."""

    st.header("🔍 Univariate Analysis")

    with st.expander("📈 Descriptive Statistics", expanded=True):
        st.subheader("Statistical Summary of Numerical Columns")
        st.dataframe(df.describe().T.style.format("{:.2f}"), use_container_width=True)
        st.info("This table provides key statistics like mean, standard deviation, min, max, and quartiles for numerical columns.")

    with st.expander("📊 Distributions of Numerical Columns", expanded=True):
        st.subheader("Histograms for All Numerical Columns")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
            st.info("No numerical columns found for histogram plotting.")
        else:
            num_plots_per_row = 3
            cols = st.columns(num_plots_per_row)
            for i, col in enumerate(numeric_cols):
                with cols[i % num_plots_per_row]:
                    fig = px.histogram(df, x=col, title=f"Distribution of {col}", marginal="box")
                    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📦 Outlier Detection (Boxplots)", expanded=True):
        st.subheader("Visualize Outliers with Boxplots")
        numeric_df = df.select_dtypes(include=np.number)
        
        if not numeric_df.empty:
            selected_outlier_cols = st.multiselect(
                "Select numeric columns to check for outliers", 
                numeric_df.columns.tolist(), 
                key="outlier_cols_select"
            )
            
            if selected_outlier_cols:
                # Custom outlier definition using IQR
                iqr_multiplier = st.slider(
                    "IQR Multiplier for Outlier Threshold",
                    min_value=1.0, max_value=3.0, value=1.5, step=0.1,
                    help="Adjust the sensitivity of outlier detection. 1.5 is standard, lower values flag more outliers."
                )

                for col in selected_outlier_cols:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - iqr_multiplier * IQR
                    upper_bound = Q3 + iqr_multiplier * IQR
                    
                    outliers_count = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
                    
                    st.markdown(f"#### Boxplot for `{col}`")
                    st.info(f"Outliers detected (outside {iqr_multiplier}*IQR): **{outliers_count}**")
                    fig = px.box(df, y=col, title=f"Boxplot of {col}")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Please select at least one numerical column to visualize outliers.")
        else:
            st.info("No numeric columns found for outlier detection.")

    with st.expander("🗂️ Categorical Feature Analysis", expanded=True):
        st.subheader("Value Counts and Bar Charts for Categorical Columns")
        categorical_cols = df.select_dtypes(include='object').columns.tolist()
        
        if not categorical_cols:
            st.info("No categorical columns found for analysis.")
        else:
            for col in categorical_cols:
                st.markdown(f"##### Value Counts for `{col}`")
                value_counts = df[col].value_counts().reset_index()
                value_counts.columns = [col, 'Count'] # Rename columns
                st.dataframe(value_counts, use_container_width=True)

                fig = px.bar(value_counts, x=col, y="Count", title=f"Distribution of {col}")
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("---")


def display_tab3(df):

    """Displays the third tab of the Data Analyzer with bivariate analysis and visualizations."""

    st.header("🤝 Bivariate Analysis")

    with st.expander("🔗 Correlation Heatmap (Numerical Features)", expanded=True):
        st.subheader("Identify Relationships Between Numerical Variables")
        numeric_df = df.select_dtypes(include=np.number)
        if not numeric_df.empty and numeric_df.shape[1] > 1:
            corr = numeric_df.corr()
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5, ax=ax)
            ax.set_title("Correlation Matrix of Numerical Features")
            st.pyplot(fig)
            st.info("Values closer to 1 or -1 indicate stronger linear relationships. Values near 0 indicate weak or no linear relationship.")
        elif numeric_df.shape[1] <= 1:
            st.info("Need at least two numerical columns to compute correlation.")
        else:
            st.info("No numeric columns found for correlation analysis.")
            
    with st.expander("Scatter Plots (Numerical vs. Numerical)", expanded=True):
        st.subheader("Explore Relationships Between Two Numerical Variables")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        if len(numeric_cols) >= 2:
            col_x = st.selectbox("Select X-axis Column", numeric_cols, key="scatter_x")
            col_y = st.selectbox("Select Y-axis Column", [c for c in numeric_cols if c != col_x], key="scatter_y")
            color_by_col = st.selectbox("Color points by (Optional)", ['None'] + df.columns.tolist(), key="scatter_color")

            if col_x and col_y:
                if color_by_col != 'None':
                    fig = px.scatter(df, x=col_x, y=col_y, color=color_by_col, 
                                     title=f"Scatter Plot of {col_x} vs {col_y} (Colored by {color_by_col})")
                else:
                    fig = px.scatter(df, x=col_x, y=col_y, title=f"Scatter Plot of {col_x} vs {col_y}")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least two numerical columns for scatter plots.")

    with st.expander("Box Plots (Numerical vs. Categorical)", expanded=True):
        st.subheader("Compare Numerical Distributions Across Categories")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include='object').columns.tolist()

        if numeric_cols and categorical_cols:
            num_col = st.selectbox("Select Numerical Column", numeric_cols, key="box_num_col")
            cat_col = st.selectbox("Select Categorical Column", categorical_cols, key="box_cat_col")
            
            if num_col and cat_col:
                fig = px.box(df, x=cat_col, y=num_col, title=f"Distribution of {num_col} by {cat_col}")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least one numerical and one categorical column for these plots.")

def display_tab4(df):

    """Displays the fourth tab of the Data Analyzer with advanced visualizations and custom plot building."""

    st.header("✨ Advanced Visualizations")

    with st.expander("Time Series Analysis (Requires Date Column)", expanded=True):
        st.subheader("Visualize Trends Over Time")
        # Attempt to identify a date column
        date_cols = [col for col in df.columns if 'date' in col.lower() or pd.api.types.is_datetime64_any_dtype(df[col])]
        
        if date_cols:
            date_col = st.selectbox("Select a Date Column", date_cols, key="date_col_select")
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

            if date_col and numeric_cols:
                st.markdown(f"**Selected Date Column:** `{date_col}`")
                df_time = df.copy()
                df_time[date_col] = pd.to_datetime(df_time[date_col], errors='coerce')
                df_time.dropna(subset=[date_col], inplace=True) # Drop rows where date conversion failed

                if not df_time.empty:
                    agg_col = st.selectbox("Select a Numerical Column for Trend", numeric_cols, key="agg_col_select")
                    time_agg = st.selectbox("Aggregate by:", ["Day", "Week", "Month", "Year"], key="time_agg_select")

                    if time_agg == "Day":
                        df_time['Period'] = df_time[date_col].dt.to_period('D').astype(str)
                    elif time_agg == "Week":
                        df_time['Period'] = df_time[date_col].dt.to_period('W').astype(str)
                    elif time_agg == "Month":
                        df_time['Period'] = df_time[date_col].dt.to_period('M').astype(str)
                    elif time_agg == "Year":
                        df_time['Period'] = df_time[date_col].dt.to_period('Y').astype(str)
                    
                    if agg_col:
                        trend_df = df_time.groupby('Period')[agg_col].sum().reset_index()
                        trend_df['Sort_Key'] = pd.to_datetime(trend_df['Period'])
                        trend_df = trend_df.sort_values('Sort_Key')

                        fig = px.line(trend_df, x='Period', y=agg_col, 
                                      title=f"Trend of {agg_col} Over Time ({time_agg} Aggregation)", markers=True)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No valid date entries found after converting the selected date column.")
            else:
                st.info("Please select a numerical column for trend analysis.")
        else:
            st.info("No columns identified as date columns in your dataset. Cannot perform time series analysis.")

    with st.expander("Custom Plot Builder", expanded=True):
        st.subheader("Create Your Own Plot")
        all_cols = df.columns.tolist()
        
        plot_type = st.selectbox("Select Plot Type", ["Bar", "Histogram", "Scatter", "Line", "Box"], key="custom_plot_type")
        
        if plot_type == "Bar":
            x_bar = st.selectbox("X-axis (Categorical)", df.select_dtypes(include='object').columns.tolist() + df.select_dtypes(include=np.number).columns.tolist(), key="x_bar")
            y_bar = st.selectbox("Y-axis (Numerical, e.g., Count or Sum)", df.select_dtypes(include=np.number).columns.tolist(), key="y_bar")
            if x_bar and y_bar:
                fig = px.bar(df, x=x_bar, y=y_bar, title=f"{y_bar} by {x_bar}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please select valid X and Y columns for the bar chart.")
        
        elif plot_type == "Histogram":
            hist_col = st.selectbox("Column to Plot", df.select_dtypes(include=np.number).columns.tolist(), key="hist_col")
            if hist_col:
                fig = px.histogram(df, x=hist_col, title=f"Distribution of {hist_col}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please select a numerical column for the histogram.")

        elif plot_type == "Scatter":
            x_scatter = st.selectbox("X-axis", df.select_dtypes(include=np.number).columns.tolist(), key="x_scatter")
            y_scatter = st.selectbox("Y-axis", df.select_dtypes(include=np.number).columns.tolist(), key="y_scatter")
            color_scatter = st.selectbox("Color by (Optional)", ['None'] + all_cols, key="color_scatter")
            if x_scatter and y_scatter:
                if color_scatter != 'None':
                    fig = px.scatter(df, x=x_scatter, y=y_scatter, color=color_scatter, title=f"Scatter of {x_scatter} vs {y_scatter}")
                else:
                    fig = px.scatter(df, x=x_scatter, y=y_scatter, title=f"Scatter of {x_scatter} vs {y_scatter}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please select valid X and Y columns for the scatter plot.")

        elif plot_type == "Line":
            x_line = st.selectbox("X-axis (Time/Order)", all_cols, key="x_line")
            y_line = st.selectbox("Y-axis (Numerical)", df.select_dtypes(include=np.number).columns.tolist(), key="y_line")
            if x_line and y_line:
                fig = px.line(df, x=x_line, y=y_line, title=f"Line Plot of {y_line} Over {x_line}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please select valid X and Y columns for the line plot.")

        elif plot_type == "Box":
            x_box = st.selectbox("X-axis (Categorical or None)", ['None'] + df.select_dtypes(include='object').columns.tolist(), key="x_box")
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
    
    """Displays the fifth tab of the Data Analyzer with options to download the processed dataset."""

    st.header("📥 Download Data")
    st.markdown("Download the currently analyzed and potentially cleaned dataset.")

    # Convert DataFrame to CSV
    # Ensure the dataframe used for download reflects any cleaning/imputation done
    current_df_for_download = st.session_state.get('df', df) # Use session state if updated, else original df
    csv_export = current_df_for_download.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="Download Processed CSV",
        data=csv_export,
        file_name="processed_data.csv",
        mime="text/csv",
        key="download_processed_csv"
    )
    st.info("This CSV includes any duplicates removed or missing values imputed during the analysis.")