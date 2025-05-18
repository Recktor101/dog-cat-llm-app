import streamlit as st
from PIL import Image
from model import predict_image
from description_model import get_breed_description
import io
import base64

# Resize image while maintaining aspect ratio
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

# Custom styles: black top bar + upload button styling
st.markdown(
    """
    <style>
    /* Top black bar */
    .top-bar {
        background-color: #000000;
        color: white;
        text-align: center;
        padding: 12px 0;
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }

    /* Style uploader */
    section[data-testid="stFileUploader"] > div {
        background-color: #000000;
        color: white;
        border-radius: 8px;
        padding: 12px;
    }

    section[data-testid="stFileUploader"] label {
        color: white !important;
    }

    section[data-testid="stFileUploader"] svg {
        fill: white !important;
    }

    /* Status text style */
    .status-text {
        text-align: center;
        font-weight: normal;
        color: #444444;
        margin-bottom: 15px;
        font-size: 14px;
        font-style: italic;
    }
    </style>

    <div class="top-bar">
        LLM at Scale — Dog & Cat Classifier App
    </div>
    """,
    unsafe_allow_html=True,
)

# Logo and title section
st.markdown(
    """
    <div style="text-align: center; margin-bottom: 5px;">
        <img src="https://raw.githubusercontent.com/Recktor101/dog-cat-llm-app/main/assets/llmatscale-logo.png" width="250">
        <div style="font-size: 10px; color: gray; margin-top: 4px;">
            LLM at Scale
        </div>
        <div style="font-size: 24px; font-weight: 700; color: black; margin-top: 8px;">
            Dog and Cat Image Classifier
        </div>
    </div>
    <hr style="margin-top: 10px; margin-bottom: 20px; border-color: black;">
    """,
    unsafe_allow_html=True,
)

# Image upload section
uploaded_file = st.file_uploader("Upload an image of a dog or cat", type=["jpg", "jpeg", "png"])

# Process image if uploaded
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image = resize_with_aspect_ratio(image, max_size=300)

    # Convert image to base64 for inline display
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    encoded_img = base64.b64encode(img_bytes).decode()

    # Display the image
    st.markdown(
        f"""
        <div style="text-align: center;">
            <img src="data:image/png;base64,{encoded_img}" style="margin: auto;" />
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Status message
    st.markdown('<div class="status-text">Classifying the image...</div>', unsafe_allow_html=True)

    # Run prediction
    label, breed_name, confidence = predict_image(image)
    animal_label = label.capitalize()

    # Display results
    st.markdown(f"**Animal:** {animal_label}")
    st.markdown(f"**Breed:** {breed_name} ({confidence:.2%} confidence)")

    # Generate breed description
    st.markdown('<div class="status-text">Generating breed description...</div>', unsafe_allow_html=True)
    description = get_breed_description(label.lower(), breed_name)
    st.markdown(f"**Breed Description:**\n\n{description}")
