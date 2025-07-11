import streamlit as st
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

@st.cache_resource
def load_model(model_name):
    """Loads a pretrained model based on the selected name."""
    with st.spinner(f"Loading {model_name} model..."):
        if model_name == "ResNet50":
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif model_name == "MobileNetV2":
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        else:
            st.error("Invalid model name selected.")
            return None
        model.eval() # Set the model to evaluation mode
    return model

@st.cache_data
def load_labels():
    """Loads ImageNet class labels."""
    import json, urllib.request
    try:
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        with urllib.request.urlopen(url) as response:
            classes = [line.decode('utf-8').strip() for line in response.readlines()]
        return classes
    except Exception as e:
        st.error(f"Could not load ImageNet labels: {e}")
        return []
    

def preprocess_image(image_input: Image.Image, size=(224, 224)):
    """Applies necessary transformations for model input."""
    # Define standard transformations for the models
    preprocess = transforms.Compose([
        transforms.Resize(size),      # Resize the image
        transforms.CenterCrop(size),  # Crop the center
        transforms.ToTensor(),        # Convert to PyTorch Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize
    ])
    img_tensor = preprocess(image_input)
    return img_tensor.unsqueeze(0) # Add a batch dimension