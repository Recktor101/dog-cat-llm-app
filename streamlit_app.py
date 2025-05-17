import streamlit as st
from PIL import Image
from model import predict_image
from description_model import get_breed_description
import base64

# Set background using base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: contain;
        background-repeat: no-repeat;
        background-position: top right;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Call the background function
set_background("llmatscale-logo.png")

# Main Streamlit App
st.set_page_config(page_title="Dog/Cat Breed Classifier", layout="centered")
st.title("🐶🐱 Dog or Cat Classifier with Breed Info")

uploaded_file = st.file_uploader("Upload an image of a dog or cat", type=["jpg", "jpeg", "png"])

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
