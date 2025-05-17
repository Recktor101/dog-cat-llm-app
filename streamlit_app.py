import streamlit as st
from PIL import Image
from model import predict_image
from description_model import get_breed_description
import io
import base64

# Set page config
st.set_page_config(page_title="Dog and Cat Image Classifier", layout="centered")

# Inject CSS for black background, white text, and styling for status messages
st.markdown(
    """
    <style>
    .main {
        background-color: black;
        color: white;
    }
    .uploader-label {
        font-size: 16px !important;
        font-weight: normal !important;
        margin-bottom: 8px !important;
        color: white !important;
    }
    input[type="file"] {
        font-size: 16px !important;
        padding: 10px !important;
        cursor: pointer;
        color: black !important;
    }
    .status-text {
        text-align: center;
        font-weight: 700;
        color: #e0e0e0;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.8);
        margin-bottom: 15px;
        font-size: 18px;
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

uploaded_file = st.file_uploader("Upload an image of a dog or cat", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    # Resize image to reduce memory usage
    image = image.resize((300, 300))

    # Convert image to base64 for embedding in HTML
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    encoded_img = base64.b64encode(img_bytes).decode()

    # Display image centered
    st.markdown(
        f"""
        <div style="text-align: center;">
            <img src="data:image/png;base64,{encoded_img}" width="300" style="margin: auto;" />
            <div style="text-align: center; font-size: 12px; color: gray; margin-bottom: 15px;">
                alien
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="status-text">Classifying the image...</div>', unsafe_allow_html=True)

    label, breed, confidence = predict_image(image)

    animal_label = label.capitalize()  # "Dog" or "Cat"

    st.markdown(f"**Animal:** {animal_label}")
    st.markdown(f"**Breed:** {breed} ({confidence:.2%} confidence)")

    if label == "DOG":
        st.markdown('<div class="status-text">Generating breed description...</div>', unsafe_allow_html=True)
        description = get_breed_description(breed)
        st.markdown(f"**Breed Description:**\n\n{description}")
