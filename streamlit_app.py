import streamlit as st
from PIL import Image
from model import predict_image
from description_model import get_breed_description

st.set_page_config(page_title="Dog/Cat Breed Classifier", layout="centered")

st.title("🐶🐱 Dog or Cat Classifier with Breed Info")
uploaded_file = st.file_uploader("Upload an image of a dog or cat", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        st.write("🔍 Classifying the image...")
        label, breed = predict_image(image)

        st.markdown(f"### Prediction: **{label}**")
        st.markdown(f"### Breed: **{breed}**")

        if label == "DOG":
            st.write("📖 Generating breed description...")
            description = get_breed_description(breed)
            st.markdown(f"**Breed Description:**\n\n{description}")
