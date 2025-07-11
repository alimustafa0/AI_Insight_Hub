import streamlit as st
import torch
from torchvision import models
from PIL import Image, ImageEnhance, ImageFilter # Added ImageFilter for new effects
import matplotlib.pyplot as plt
import wikipedia
import piexif # For more robust EXIF handling
import io
import numpy as np
import pandas as pd # Added for downloading predictions
from modules.image_classifier import load_model, load_labels, preprocess_image

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Advanced Image Classifier")
st.title("🖼️ Advanced Image Classifier")
st.markdown("Upload an image to classify it using pretrained deep learning models and explore its metadata and apply cool filters!")

# --- Load Models and Labels ---
labels = load_labels()

# --- Initialize Session State for Image and Filters ---
# This is crucial for persisting image data and filter selections across reruns
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'brightness' not in st.session_state:
    st.session_state.brightness = 1.0
if 'contrast' not in st.session_state:
    st.session_state.contrast = 1.0
if 'sharpness' not in st.session_state:
    st.session_state.sharpness = 1.0
if 'active_filters' not in st.session_state: # New: List to store active filters
    st.session_state.active_filters = []
if 'gaussian_blur_radius' not in st.session_state:
    st.session_state.gaussian_blur_radius = 2
if 'file_uploader_key' not in st.session_state: # New: Key for resetting file uploader
    st.session_state.file_uploader_key = 0


# --- Image Upload Section ---
st.subheader("⬆️ Upload Your Image")
# Use the dynamic key for the file uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "bmp"], key=f"image_uploader_{st.session_state.file_uploader_key}")

# Handle new file upload or maintain existing image
if uploaded_file:
    # Check if a new file has been uploaded compared to the one in session state
    # Compare file_id for robustness
    if st.session_state.uploaded_file is None or uploaded_file.file_id != st.session_state.uploaded_file.file_id:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.session_state.uploaded_file = uploaded_file # Store the uploaded file object
            st.session_state.processed_image = image # Initialize processed_image
            st.success("Image uploaded successfully!")
            
            # Reset filters and enhancements when a new image is uploaded
            st.session_state.brightness = 1.0
            st.session_state.contrast = 1.0
            st.session_state.sharpness = 1.0
            st.session_state.active_filters = [] # Reset active filters
            st.session_state.gaussian_blur_radius = 2
            # No rerun here, let the main flow continue to render the image
        except Exception as e:
            st.error(f"Error loading image: {e}. Please upload a valid image file.")
            st.session_state.uploaded_file = None # Clear state on error
            st.session_state.processed_image = None
    # else: if the same file is uploaded, st.session_state.uploaded_file already holds it.
    # No need to re-assign or re-open here unless specific processing is needed.
else:
    # If the file_uploader widget is empty, ensure session state is also cleared
    if st.session_state.uploaded_file is not None and uploaded_file is None:
        st.session_state.uploaded_file = None
        st.session_state.processed_image = None
        # Reset filter states as well
        st.session_state.brightness = 1.0
        st.session_state.contrast = 1.0
        st.session_state.sharpness = 1.0
        st.session_state.active_filters = []
        st.session_state.gaussian_blur_radius = 2
        st.rerun() # Rerun to clear the display if file was removed from widget


# --- Display Sections ---
if st.session_state.processed_image: # Use processed_image from session state
    st.subheader("🖼️ Image Preview & Preprocessing")
    
    # Image Enhancement Options
    st.markdown("##### Image Enhancement Options")
    st.write("Adjust brightness, contrast, sharpness, and apply artistic filters:")

    st.session_state.brightness = st.slider("Brightness", 0.1, 3.0, st.session_state.brightness, 0.1, key="brightness_slider")
    st.session_state.contrast = st.slider("Contrast", 0.1, 3.0, st.session_state.contrast, 0.1, key="contrast_slider")
    st.session_state.sharpness = st.slider("Sharpness", 0.1, 3.0, st.session_state.sharpness, 0.1, key="sharpness_slider")
    
    st.markdown("##### Filters")
    filter_options = ["Grayscale", "Sepia", "Gaussian Blur", "Edge Enhance", "Contour"]
    cols = st.columns(len(filter_options))

    # Custom CSS for filter buttons
    st.markdown("""
        <style>
        /* Base button style */
        .stButton>button {
            background-color: #f0f2f6;
            color: #333;
            border: 1px solid #ddd;
            border-radius: 0.5rem;
            padding: 0.5rem 1rem;
            margin: 0.2rem;
            cursor: pointer;
            transition: all 0.2s ease-in-out;
            font-weight: 500;
            width: 100%; /* Ensure buttons take full column width */
            text-align: center;
        }
        .stButton>button:hover {
            background-color: #e0e0e0;
        }
        /* Active filter button style */
        .filter-button-active { /* Changed class name to avoid conflict with default Streamlit button styles */
            background-color: #4CAF50 !important; /* Green for active, !important to override */
            color: white !important;
            border-color: #4CAF50 !important;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2) !important;
        }
        </style>
    """, unsafe_allow_html=True)

    for i, filter_name in enumerate(filter_options):
        with cols[i]:
            is_active = filter_name in st.session_state.active_filters
            
            # When button is clicked, toggle the filter in active_filters list
            if st.button(filter_name, key=f"filter_btn_{filter_name}"):
                if is_active:
                    st.session_state.active_filters.remove(filter_name)
                else:
                    st.session_state.active_filters.append(filter_name)
                st.rerun() # Rerun to apply filter changes and update button style

    # JavaScript to apply the 'filter-button-active' class dynamically
    # This script needs to run after Streamlit renders the buttons
    st.markdown(f"""
        <script>
            // Function to apply/remove active class
            function updateFilterButtonStyles() {{
                var buttons = window.parent.document.querySelectorAll('.stButton button');
                // Ensure activeFilters is a valid array from Python state
                var activeFilters = {st.session_state.active_filters if st.session_state.active_filters else "[]"}; 

                buttons.forEach(function(button) {{
                    var buttonText = button.textContent.trim();
                    // Only apply to filter buttons, not other buttons like "Download" or "Classify"
                    if (['Grayscale', 'Sepia', 'Gaussian Blur', 'Edge Enhance', 'Contour'].includes(buttonText)) {{
                        if (activeFilters.includes(buttonText)) {{
                            button.classList.add('filter-button-active');
                        }} else {{
                            button.classList.remove('filter-button-active');
                        }}
                    }}
                }});
            }}

            // Run on initial load and after Streamlit updates
            updateFilterButtonStyles();

            // Use a MutationObserver to re-run the styling function when DOM changes
            // This is more robust for Streamlit's dynamic rendering
            const observer = new MutationObserver(function(mutations) {{
                mutations.forEach(function(mutation) {{
                    if (mutation.type === 'childList' || mutation.type === 'attributes') {{
                        updateFilterButtonStyles();
                    }}
                }});
            }});

            // Observe the main Streamlit app container
            // Adjust this selector if your Streamlit app is nested differently
            const stApp = window.parent.document.querySelector('.stApp');
            if (stApp) {{
                // Disconnect existing observers to prevent multiple observations
                if (window.stAppObserver) {{
                    window.stAppObserver.disconnect();
                }}
                observer.observe(stApp, {{ childList: true, subtree: true, attributes: true }});
                window.stAppObserver = observer; // Store observer to disconnect later
            }}
        </script>
    """, unsafe_allow_html=True)


    if "Gaussian Blur" in st.session_state.active_filters:
        st.session_state.gaussian_blur_radius = st.slider("Blur Radius", 0, 10, st.session_state.gaussian_blur_radius, key="blur_radius")

    # Apply enhancements
    # Always start with the original image from session state for enhancements
    current_image_for_processing = Image.open(st.session_state.uploaded_file).convert("RGB") 
    enhanced_image = ImageEnhance.Brightness(current_image_for_processing).enhance(st.session_state.brightness)
    enhanced_image = ImageEnhance.Contrast(enhanced_image).enhance(st.session_state.contrast)
    enhanced_image = ImageEnhance.Sharpness(enhanced_image).enhance(st.session_state.sharpness)

    # Apply all active filters in a defined order (important for consistent results)
    for active_filter in filter_options: # Iterate through all possible filters in defined order
        if active_filter in st.session_state.active_filters: # Apply only if active
            if active_filter == "Grayscale":
                enhanced_image = enhanced_image.convert("L").convert("RGB")
            elif active_filter == "Sepia":
                sepia_image = Image.new('RGB', enhanced_image.size)
                for x in range(enhanced_image.width):
                    for y in range(enhanced_image.height):
                        r, g, b = enhanced_image.getpixel((x, y))
                        tr = int(0.393 * r + 0.769 * g + 0.189 * b)
                        tg = int(0.349 * r + 0.686 * g + 0.168 * b)
                        tb = int(0.272 * r + 0.534 * g + 0.131 * b)
                        sepia_image.putpixel((x, y), (min(255, tr), min(255, tg), min(255, tb)))
                enhanced_image = sepia_image
            elif active_filter == "Gaussian Blur":
                enhanced_image = enhanced_image.filter(ImageFilter.GaussianBlur(radius=st.session_state.gaussian_blur_radius))
            elif active_filter == "Edge Enhance":
                enhanced_image = enhanced_image.filter(ImageFilter.EDGE_ENHANCE_MORE)
            elif active_filter == "Contour":
                enhanced_image = enhanced_image.filter(ImageFilter.CONTOUR)

    st.session_state.processed_image = enhanced_image # Update the processed image in session state
    
    col1, col2 = st.columns(2) # New columns for side-by-side image display

    with col1:
        st.markdown("##### Original Image")
        st.image(Image.open(st.session_state.uploaded_file).convert("RGB"), caption="Uploaded Image", use_container_width=True) 

    with col2:
        st.markdown("##### Processed Image (for Classification)")
        st.image(st.session_state.processed_image, caption="Processed Image", use_container_width=True)

    if st.button("Download Processed Image"):
        buf = io.BytesIO()
        st.session_state.processed_image.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button(
            label="Download Image",
            data=byte_im,
            file_name="processed_image.png",
            mime="image/png"
        )

    st.markdown("---")

    st.subheader("📸 EXIF Metadata (if available)")
    exif_data = {}
    try:
        if "exif" in Image.open(st.session_state.uploaded_file).info:
            exif_bytes = Image.open(st.session_state.uploaded_file).info["exif"]
            
            if isinstance(exif_bytes, bytes):
                try:
                    exif_dict = piexif.load(exif_bytes)
                except Exception as load_e:
                    st.warning(f"piexif.load failed to process EXIF bytes: {load_e}. This might indicate corrupted or unsupported EXIF format.")
                    exif_dict = None
                
                if isinstance(exif_dict, dict):
                    for ifd_name in exif_dict:
                        if isinstance(exif_dict[ifd_name], dict):
                            for key, value in exif_dict[ifd_name].items():
                                try:
                                    tag_name = ExifTags.TAGS.get(key, key)
                                    if isinstance(value, bytes):
                                        exif_data[tag_name] = value.decode('utf-8', errors='ignore')
                                    else:
                                        exif_data[tag_name] = value
                                except Exception as inner_e:
                                    exif_data[ExifTags.TAGS.get(key, key)] = f"<Undecodable/Unprocessable Value: {type(value).__name__}>"
                        else:
                            st.warning(f"Skipping malformed EXIF IFD '{ifd_name}'. Expected dictionary, got '{type(exif_dict[ifd_name]).__name__}'.")
                else:
                    st.info(f"No parsable EXIF metadata found after loading. piexif.load returned unexpected type: {type(exif_dict).__name__}.")
            else:
                st.warning(f"Image info 'exif' is not in expected bytes format. Found: {type(exif_bytes).__name__}. Skipping EXIF parsing.")

            if exif_data:
                st.json(exif_data)
                st.info("EXIF data provides details about the camera, date, settings, etc.")
            else:
                st.info("No EXIF metadata found for this image or could not be parsed.")
        else:
            st.info("No EXIF metadata found for this image.")
    except Exception as e:
        st.warning(f"An unexpected error occurred during EXIF metadata processing: {e}")

    st.markdown("---")

    st.subheader("🧠 Model Selection & Prediction")
    model_name = st.selectbox(
        "Choose a Pretrained Model",
        ["ResNet50", "MobileNetV2"],
        help="ResNet50 is a powerful, deep model. MobileNetV2 is lighter and faster."
    )

    model = load_model(model_name)

    if model and labels:
        st.markdown(f"**Selected Model:** `{model_name}`")
        if model_name == "ResNet50":
            st.info("ResNet50 is a deep convolutional neural network known for its high accuracy on ImageNet.")
        elif model_name == "MobileNetV2":
            st.info("MobileNetV2 is a lightweight, efficient model suitable for mobile and embedded vision applications.")
        
        # Perform Classification
        if st.button("Classify Image"):
            if st.session_state.processed_image:
                with st.spinner("Classifying image... This might take a moment."):
                    img_tensor = preprocess_image(st.session_state.processed_image)
                    outputs = model(img_tensor)
                    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                    top5_prob, top5_catid = torch.topk(probabilities, 5)

                st.subheader("📌 Prediction Results")
                
                top_prediction_label = labels[top5_catid[0]]
                top_prediction_confidence = round(top5_prob[0].item(), 3) * 100
                st.success(f"**Top Prediction:** `{top_prediction_label}` with **Confidence:** `{top_prediction_confidence:.2f}%`")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                y_labels = [labels[i] for i in top5_catid]
                y_values = top5_prob.detach().numpy() * 100
                ax.barh(y=y_labels, width=y_values, color='skyblue')
                ax.set_xlabel("Confidence (%)")
                ax.set_title("Top 5 Predictions")
                ax.invert_yaxis()
                st.pyplot(fig)
                plt.close(fig)

                st.markdown("---")
                predictions_df = pd.DataFrame({
                    "Rank": range(1, 6),
                    "Label": y_labels,
                    "Confidence (%)": y_values
                })
                csv_export = predictions_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Predictions CSV",
                    data=csv_export,
                    file_name="image_predictions.csv",
                    mime="text/csv",
                    key="download_predictions_btn"
                )


                st.subheader("📚 Learn More About the Prediction")
                
                wiki_summary_placeholder = st.empty()
                with st.spinner(f"Fetching Wikipedia info for '{top_prediction_label}'..."):
                    try:
                        summary = wikipedia.summary(top_prediction_label, sentences=3, auto_suggest=False)
                        wiki_summary_placeholder.markdown(f"**Wikipedia Summary for {top_prediction_label}:**")
                        wiki_summary_placeholder.info(summary)
                        wiki_summary_placeholder.markdown(f"[Read more on Wikipedia](https://en.wikipedia.org/wiki/{top_prediction_label.replace(' ', '_')})")
                    except wikipedia.exceptions.PageError:
                        try:
                            summary = wikipedia.summary(top_prediction_label, sentences=3, auto_suggest=True)
                            wiki_summary_placeholder.markdown(f"**Wikipedia Summary for {top_prediction_label}:**")
                            wiki_summary_placeholder.info(summary)
                            wiki_summary_placeholder.markdown(f"[Read more on Wikipedia](https://en.wikipedia.org/wiki/{wikipedia.search(top_prediction_label)[0].replace(' ', '_')})")
                        except Exception:
                            wiki_summary_placeholder.warning(f"Could not find a Wikipedia summary for '{top_prediction_label}'.")
                    except Exception as e:
                        wiki_summary_placeholder.warning(f"An error occurred while fetching Wikipedia info: {e}")
            else:
                st.warning("Please upload an image first to classify.")
    else:
        st.warning("Model or labels could not be loaded. Please check your internet connection or try again.")

else:
    st.info("Upload an image using the file uploader above to start the analysis.")

# --- Clear Image Button (always visible if an image is uploaded) ---
if st.session_state.uploaded_file is not None:
    if st.button("🗑️ Clear Uploaded Image", key="clear_image_button"):
        st.session_state.uploaded_file = None
        st.session_state.processed_image = None
        st.session_state.brightness = 1.0
        st.session_state.contrast = 1.0
        st.session_state.sharpness = 1.0
        st.session_state.active_filters = []
        st.session_state.gaussian_blur_radius = 2
        st.session_state.file_uploader_key += 1 # Increment key to force reset of file_uploader widget
        st.success("Image cleared. Upload a new one!")
        st.rerun()
