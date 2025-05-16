import streamlit as st
from PIL import Image
from model import predict_breed
from description_model import generate_description

st.set_page_config(page_title="Dog/Cat Breed Classifier")

st.title("🐶 Dog Breed Identifier + Description")

uploaded_file = st.file_uploader("Upload a dog image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Classifying..."):
        breed, confidence = predict_breed(image)
        description = generate_description(breed)

    st.markdown(f"### ✅ Predicted Breed: `{breed}`")
    st.markdown(f"**Confidence:** `{confidence*100:.2f}%`")
    st.markdown("### 📄 Description")
    st.write(description)
