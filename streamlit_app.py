import streamlit as st
from PIL import Image
from model import predict_image
from description_model import get_breed_description
import base64

# Function to add the background image
def set_background(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    background_style = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(background_style, unsafe_allow_html=True)

# Set Streamlit page settings
st.set_page_config(page_title="Dog/Cat Breed Classifier", layout="centered")

# Set the logo as background
set_background("assets/llmatscale-logo.png")

# Title
st.title("🐶🐱 Dog or Cat Classifier with Breed Info")

# File uploader
uploaded_file = st.file_uploader("Upload an image of a dog or cat", type=["jpg", "jpeg", "png"])

# Main logic
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        st.write("🔍 Classifying the image...")
        label, breed, confidence = predict_image(image)

        st.markdown(f"### Prediction: **{label}**")
        st.markdown(f"### Breed: **{breed}**")
        st.markdown(f"### Confidence: **{confidence:.2%}**")

        if label == "DOG":
            st.write("📖 Generating breed description...")
            description = get_breed_description(breed)
            st.markdown(f"**Breed Description:**\n\n{description}")
