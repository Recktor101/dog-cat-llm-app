import streamlit as st
from model import predict_breed

st.title("Dog & Cat Breed Classifier with Description")

uploaded_file = st.file_uploader("Upload a dog or cat image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    breed, description, confidence = predict_breed("temp.jpg")

    st.image("temp.jpg", caption=f"Uploaded Image", use_column_width=True)
    st.markdown(f"### Prediction: {breed.replace('_', ' ')}")
    st.markdown(f"**Confidence:** {confidence:.2f}%")
    st.write(description)
