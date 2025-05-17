import streamlit as st
from PIL import Image
from model import predict_image
from description_model import get_breed_description

# Set page config
st.set_page_config(page_title="Dog/Cat Classifier", layout="centered")

# Centered logo at top
st.markdown(
    """
    <div style="text-align:center;">
        <img src="https://raw.githubusercontent.com/your-username/your-repo/main/assets/llmatscale-logo.png" width="250"/>
    </div>
    """,
    unsafe_allow_html=True,
)

# Optional subheading like in example
st.markdown(
    """
    <h3 style="text-align:center;">Dog/Cat Image Classifier with Breed Prediction</h3>
    <p style="text-align:center;"><a href="#">GEN AI Bootcamp 2025</a></p>
    <hr>
    """,
    unsafe_allow_html=True,
)

# File uploader
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
