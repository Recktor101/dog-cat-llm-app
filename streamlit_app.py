import streamlit as st
from model import predict_breed
from PIL import Image
import os

st.title("Dog/Cat Breed Classifier & Description")

uploaded_file = st.file_uploader("Upload a dog or cat image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Save temporarily
    temp_path = "temp.jpg"
    img.save(temp_path)

    with st.spinner("Classifying..."):
        breed, description, confidence = predict_breed(temp_path)
        st.subheader(f"Prediction: {breed.replace('_', ' ')}")
        st.write(description)
        st.write(f"Confidence: {confidence:.2f}%")

    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)
