import streamlit as st
from PIL import Image

from model import predict_dog_breed
from description_model import generate_breed_description

st.title("🐶 Dog Breed Classifier & Info")

uploaded_file = st.file_uploader("Upload a dog image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.subheader("Prediction:")
    breed = predict_dog_breed(image)
    st.success(f"Breed: {breed}")

    st.subheader("Breed Description:")
    description = generate_breed_description(breed)
    st.info(description)
