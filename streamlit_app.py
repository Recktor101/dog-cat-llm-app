import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from PIL import Image
import requests
from io import BytesIO

# Load the FLAN-T5 model and tokenizer
def load_flan_t5_model():
    model_name = "google/flan-t5-small"  # Pretrained FLAN-T5 model
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model.eval()  # Set the model to evaluation mode
    return model, tokenizer

# Generate a description using FLAN-T5
def generate_description(model, tokenizer, text_input):
    inputs = tokenizer(text_input, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(inputs['input_ids'], max_length=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Function to process the uploaded image
def process_image(uploaded_image):
    image = Image.open(uploaded_image)
    return image

# Streamlit app
st.title("Dog and Cat Image Classifier with FLAN-T5 Description")

# Upload image
uploaded_image = st.file_uploader("Upload a Dog or Cat Image", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    # Display uploaded image
    image = process_image(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load FLAN-T5 model and tokenizer
    flan_t5_model, flan_t5_tokenizer = load_flan_t5_model()

    # Generate description using the model
    text_input = "Describe a dog or cat image."  # Adjust this for better results based on your use case
    description = generate_description(flan_t5_model, flan_t5_tokenizer, text_input)
    
    # Display description
    st.subheader("Generated Description:")
    st.write(description)
