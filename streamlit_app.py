import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from PIL import Image
import requests
from io import BytesIO

# Load the FLAN-T5 model and tokenizer from Hugging Face
@st.cache_resource
def load_flan_t5_model():
    model_name = "google/flan-t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return model, tokenizer

# Function to process the image (here you can customize based on your need)
def classify_image(image):
    # Dummy function: You can replace this with actual model classification logic
    return "dog" if image else "cat"

# Generate description using FLAN-T5 model
def generate_description(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(inputs.input_ids)
    description = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return description

# Streamlit App UI
st.title("Dog vs. Cat Image Classifier + Description Generator")

st.write(
    "Upload a picture of a dog or cat, and the model will classify it and provide a description."
)

# Upload image
image = st.file_uploader("Upload an image of a dog or cat", type=["jpg", "png", "jpeg"])

if image is not None:
    # Open and display the image
    img = Image.open(image)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Load model and tokenizer
    model, tokenizer = load_flan_t5_model()

    # Image classification logic
    classification = classify_image(img)
    st.write(f"This is a {classification}.")

    # Generate description based on classification
    description = generate_description(f"Describe a {classification}.", model, tokenizer)
    st.write("Description:", description)
