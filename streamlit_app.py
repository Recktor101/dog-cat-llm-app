import streamlit as st
from PIL import Image
from model import predict_image
from description_model import get_breed_description

# Set page config
st.set_page_config(page_title="Dog and Cat Image Classifier", layout="centered")

# Inject CSS for black background, white text, and slightly bigger uploader styles
st.markdown(
    """
    <style>
    .main {
        background-color: black;
        color: white;
    }
    /* Slightly bigger uploader label text, normal weight */
    .css-1d391kg p {
        font-size: 16px !important;
        font-weight: normal !important;
        margin-bottom: 8px !important;
        color: white !important;
    }
    /* Slightly larger file input button */
    input[type="file"] {
        font-size: 16px !important;
        padding: 10px !important;
        cursor: pointer;
        color: black !important; /* ensure visible text */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header with centered logo and white title text
st.markdown(
    """
    <div style="text-align: center; margin-bottom: 5px;">
        <img src="https://raw.githubusercontent.com/Recktor101/dog-cat-llm-app/main/assets/llmatscale-logo.png" width="250">
        <div style="font-size: 10px; color: lightgray; margin-top: 4px;">
            LLM at Scale
        </div>
        <div style="font-size: 24px; font-weight: 700; color: #FFFFFF; margin-top: 8px;">
            Dog and Cat Image Classifier
        </div>
    </div>
    <hr style="margin-top: 10px; margin-bottom: 20px; border-color: white;">
    """,
    unsafe_allow_html=True,
)

# Just use the default uploader label (no extra markdown) for simplicity
uploaded_file = st.file_uploader("Upload an image of a dog or cat", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.markdown(
        """
        <div style="text-align: center; font-size: 12px; color: gray; margin-bottom: 15px;">
            alien
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Predict"):
        st.write("Classifying the image...")
        label, breed, confidence = predict_image(image)

        st.markdown(f"**Prediction:** {label}")
        st.markdown(f"**Breed:** {breed}")
        st.markdown(f"**Confidence:** {confidence:.2%}")

        if label == "DOG":
            st.write("Generating breed description...")
            description = get_breed_description(breed)
            st.markdown(f"**Breed Description:**\n\n{description}")
