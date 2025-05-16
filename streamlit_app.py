import streamlit as st
from model import predict_breed
from PIL import Image

st.title("Dog and Cat Breed Classifier with Description")

uploaded_file = st.file_uploader("Upload an image of a dog or cat", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded image temporarily
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image("temp.jpg", caption="Uploaded Image", use_column_width=True)

    try:
        breed, description, confidence = predict_breed("temp.jpg")

        st.subheader(f"Prediction: {breed.replace('_', ' ')}")
        st.write(f"Confidence: {confidence*100:.2f}%")
        st.write("Breed Description:")
        st.write(description)

    except Exception as e:
        st.error(f"Error: {str(e)}")
