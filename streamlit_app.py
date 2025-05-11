import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
from PIL import Image
import torch
from torchvision import models, transforms

# Load the Flan-T5 model and tokenizer
@st.cache_resource
def load_flan_t5_model():
    model_name = "google/flan-t5-small"  # Change model as needed
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return model, tokenizer

# Load the pretrained ResNet model for image classification
@st.cache_resource
def load_image_classifier():
    model = models.resnet18(pretrained=True)
    model.eval()
    return model

# Preprocess the image before passing it to the model
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0)

# Function to get class labels (using ImageNet)
def get_class_labels():
    # ImageNet class labels are available online, we use them here
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json'
    import json
    import requests
    response = requests.get(url)
    return json.loads(response.text)

# Streamlit UI Setup
st.title("Dog vs Cat Classifier with Description")

# Load models
flan_t5_model, flan_t5_tokenizer = load_flan_t5_model()
image_classifier = load_image_classifier()
class_labels = get_class_labels()

# Upload Image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess image and get predictions
    image_tensor = preprocess_image(img)
    with torch.no_grad():
        outputs = image_classifier(image_tensor)
        _, predicted_class = torch.max(outputs, 1)
        
    # Get class label and associated description
    predicted_label = class_labels[str(predicted_class.item())][1]
    
    # Generate description using FLAN-T5
    input_text = f"Describe a {predicted_label}."
    inputs = flan_t5_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    summary_ids = flan_t5_model.generate(inputs['input_ids'], max_length=100, num_beams=5, early_stopping=True)
    description = flan_t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    st.subheader(f"Prediction: {predicted_label}")
    st.write(f"Description: {description}")
