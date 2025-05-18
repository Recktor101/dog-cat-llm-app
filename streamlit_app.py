import streamlit as st
from PIL import Image
import io
import base64

def resize_with_aspect_ratio(image, max_size=300):
    w, h = image.size
    if w > h:
        new_w = max_size
        new_h = int(h * max_size / w)
    else:
        new_h = max_size
        new_w = int(w * max_size / h)
    return image.resize((new_w, new_h))

st.set_page_config(page_title="Dog and Cat Image Classifier", layout="centered")

st.markdown(
    """
    <style>
    /* Make whole file uploader black with white text */
    div[data-testid="stFileUploader"] > div:first-child {
        background-color: black !important;
        color: white !important;
        border: 2px dashed white !important;
        border-radius: 12px;
        padding: 40px 20px;
        text-align: center;
        font-weight: 700;
        font-size: 18px;
        cursor: pointer;
        position: relative;
        user-select: none;
    }
    /* White text for label inside uploader */
    div[data-testid="stFileUploader"] label {
        color: white !important;
        font-weight: 700 !important;
    }
    /* Style browse button: black background, white text, white border */
    div[data-testid="stFileUploader"] button {
        background-color: black !important;
        color: white !important;
        border: 2px solid white !important;
        padding: 8px 20px !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        margin-top: 15px;
        cursor: pointer;
    }
    /* Browse button hover effect */
    div[data-testid="stFileUploader"] button:hover {
        background-color: #222 !important;
        border-color: #ddd !important;
    }
    /* Hide native file input but keep it clickable */
    div[data-testid="stFileUploader"] input[type="file"] {
        opacity: 0;
        width: 100%;
        height: 100%;
        position: absolute;
        top: 0; left: 0;
        cursor: pointer;
        z-index: 10;
    }
    /* White color for icon */
    div[data-testid="stFileUploader"] svg {
        fill: white !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader("Upload an image of a dog or cat", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image = resize_with_aspect_ratio(image, max_size=300)

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

    st.markdown('<div style="text-align:center; font-weight: 600; color: black; margin-top: 10px;">Classifying the image...</div>', unsafe_allow_html=True)

    # For demo: replace with actual predict_image call
    label, breed_name, confidence = "dog", "Golden Retriever", 0.95

    animal_label = label.capitalize()

    st.markdown(f'<div style="font-weight: 600; color: black; margin-top: 10px;"><strong>Animal:</strong> {animal_label}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-weight: 600; color: black; margin-top: 5px;"><strong>Breed:</strong> {breed_name} ({confidence:.2%} confidence)</div>', unsafe_allow_html=True)

    st.markdown('<div style="text-align:center; font-style: italic; color: #444; margin-top: 15px;">Generating breed description...</div>', unsafe_allow_html=True)

    # For demo: replace with actual get_breed_description call
    description = "The Golden Retriever is a friendly, intelligent dog breed that loves people."
    st.markdown(f'<div style="font-weight: 600; color: black; margin-top: 10px;"><strong>Breed Description:</strong><br><br>{description}</div>', unsafe_allow_html=True)
