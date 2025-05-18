import streamlit as st
from PIL import Image
from model import predict_image
from description_model import get_breed_description
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
    /* Entire uploader container */
    div[data-testid="stFileUploader"] > div:first-child {
        background-color: black !important;
        border: 2px dashed white !important;
        border-radius: 10px;
        padding: 40px 20px;
        text-align: center;
        font-size: 18px;
        font-weight: 700;
        color: white !important;
        cursor: pointer;
        position: relative;
        user-select: none;
    }

    /* Text inside uploader including placeholder text */
    div[data-testid="stFileUploader"] label,
    div[data-testid="stFileUploader"] span {
        color: white !important;
        font-weight: 700 !important;
        pointer-events: none;
    }

    /* The browse files button */
    div[data-testid="stFileUploader"] button {
        background-color: black !important;
        color: white !important;
        border: 2px solid white !important;
        padding: 8px 16px !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        cursor: pointer !important;
        margin-top: 12px;
    }
    div[data-testid="stFileUploader"] button:hover {
        background-color: #222 !important;
        border-color: #ddd !important;
    }

    /* The file input itself is invisible but clickable */
    div[data-testid="stFileUploader"] input[type="file"] {
        opacity: 0;
        width: 100%;
        height: 100%;
        position: absolute;
        top: 0; left: 0;
        cursor: pointer;
        z-index: 10;
    }

    /* Icons inside uploader */
    div[data-testid="stFileUploader"] svg {
        fill: white !important;
        color: white !important;
    }

    /* Make all text in app black and bold */
    body, 
    .css-18e3th9,
    .css-1d391kg,
    .stMarkdown,
    div,
    p,
    span {
        color: #000000 !important;
        font-weight: 600 !important;
    }

    /* Response text styling */
    .response-text {
        color: #000000;
        font-weight: 600;
        margin-top: 10px;
        margin-bottom: 10px;
        font-size: 16px;
    }

    /* Status text */
    .status-text {
        text-align: center;
        font-weight: normal;
        color: #444444;
        margin-bottom: 15px;
        font-size: 14px;
        font-style: italic;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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

    st.markdown('<div class="status-text">Classifying the image...</div>', unsafe_allow_html=True)

    label, breed_name, confidence = predict_image(image)
    animal_label = label.capitalize()

    st.markdown(f'<div class="response-text"><strong>Animal:</strong> {animal_label}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="response-text"><strong>Breed:</strong> {breed_name} ({confidence:.2%} confidence)</div>', unsafe_allow_html=True)

    st.markdown('<div class="status-text">Generating breed description...</div>', unsafe_allow_html=True)

    description = get_breed_description(label.lower(), breed_name)
    st.markdown(f'<div class="response-text"><strong>Breed Description:</strong><br><br>{description}</div>', unsafe_allow_html=True)
