import streamlit as st
from model import predict_breed
from PIL import Image

st.title("Dog & Cat Breed Classifier 🐶🐱")

uploaded_file = st.file_uploader("Upload a dog or cat image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Save image temporarily
    image.save("temp.jpg")

    with st.spinner("Classifying..."):
        breed, description, confidence = predict_breed("temp.jpg")
        st.markdown(f"### Prediction: {breed.replace('_', ' ')}")
        st.markdown(f"**Confidence:** {confidence*100:.2f}%")
        st.markdown(f"**Description:** {description}")
