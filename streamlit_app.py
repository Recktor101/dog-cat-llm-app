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
    /* Entire file uploader area */
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
        user-select: none;
        position: relative;
    }
    
    /* Make placeholder text white */
    div[data-testid="stFileUploader"] label {
        color: white !important;
        font-weight: 700 !important;
    }
    
    /* Browse button */
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
    div[data-testid="stFileUploader"] button:hover {
        background-color: #222 !important;
        border-color: #ddd !important;
    }
    
    /* Hide default file input */
    div[data-testid="stFileUploader"] input[type="file"] {
        opacity: 0;
        width: 100%;
        height: 100%;
        position: absolute;
        top: 0; left: 0;
        cursor: pointer;
        z-index: 10;
    }
    
    /* Icon color */
    div[data-testid="stFileUploader"] svg {
        fill: white !important;
        color: white !important;
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

    st.markdown('<div style="text-align:center; font-weight: 600; color: black; margin-top: 10px;">Classifying the image...</div>', unsafe_allow_html=True)

    label, breed_name, confidence = predict_image(image)
    animal_label = label.capitalize()

    st.markdown(f'<div style="font-weight: 600; color: black; margin-top: 10px;"><strong>Animal:</strong> {animal_label}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-weight: 600; color: black; margin-top: 5px;"><strong>Breed:</strong> {breed_name} ({confidence:.2%} confidence)</div>', unsafe_allow_html=True)

    st.markdown('<div style="text-align:center; font-style: italic; color: #444; margin-top: 15px;">Generating breed description...</div>', unsafe_allow_html=True)

    description = get_breed_description(label.lower(), breed_name)
    st.markdown(f'<div style="font-weight: 600; color: black; margin-top: 10px;"><strong>Breed Description:</strong><br><br>{description}</div>', unsafe_allow_html=True)
