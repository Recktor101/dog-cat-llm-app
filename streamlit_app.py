import streamlit as st
from PIL import Image
from model import predict_image
from description_model import get_breed_description
import io
import base64

# Resize helper to keep proportions
def resize_with_aspect_ratio(image, max_size=300):
    w, h = image.size
    if w > h:
        new_w = max_size
        new_h = int(h * max_size / w)
    else:
        new_h = max_size
        new_w = int(w * max_size / h)
    return image.resize((new_w, new_h))

# Set page config
st.set_page_config(page_title="Dog and Cat Image Classifier", layout="centered")

# Custom CSS for dark theme and styling
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
        font-weight: normal;
        color: #555555;  /* Dark gray */
        margin-bottom: 15px;
        font-size: 14px; /* Smaller font size */
        font-style: italic;  /* Optional: italic for subtle effect */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header with logo and title
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
    # Open and resize image
    image = Image.open(uploaded_file).convert("RGB")
    image = resize_with_aspect_ratio(image, max_size=300)

    # Display image
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    encoded_img = base64.b64encode(img_bytes).decode()

    st.markdown(
        f"""
        <div style="text-align: center;">
            <img src="data:image/png;base64,{encoded_img}" style="margin: auto;" />
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Show status while classifying
    st.markdown('<div class="status-text">Classifying the image...</div>', unsafe_allow_html=True)

    # Predict animal type, breed and confidence
    label, breed_name, confidence = predict_image(image)
    animal_label = label.capitalize()

    st.markdown(f"**Animal:** {animal_label}")
    st.markdown(f"**Breed:** {breed_name} ({confidence:.2%} confidence)")

    # Generate and show breed description
    st.markdown('<div class="status-text">Generating breed description...</div>', unsafe_allow_html=True)
    description = get_breed_description(label.lower(), breed_name)
    st.markdown(f"**Breed Description:**\n\n{description}")
