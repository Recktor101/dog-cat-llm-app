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
    /* Remove top black bar entirely (no styles for it) */

    /* Hide the actual file input button */
    div[data-testid="stFileUploader"] input[type="file"] {
        opacity: 0;
        position: absolute;
        width: 100%;
        height: 100%;
        cursor: pointer;
        z-index: 2;
    }

    /* Style the uploader container */
    div[data-testid="stFileUploader"] > div:first-child {
        background-color: black !important;
        color: white !important;
        border: 2px dashed white !important;
        border-radius: 10px;
        padding: 20px;
        position: relative;
        text-align: center;
        font-weight: 600;
        cursor: pointer;
        font-size: 16px;
        user-select: none;
    }

    /* Style the label that contains the "browse files" text */
    div[data-testid="stFileUploader"] label {
        color: white !important;
        cursor: pointer;
        font-weight: 600;
    }

    /* Style SVG icon inside uploader */
    div[data-testid="stFileUploader"] svg {
        color: white !important;
        fill: white !important;
        vertical-align: middle;
    }

    /* Hover effect for uploader */
    div[data-testid="stFileUploader"] > div:first-child:hover {
        background-color: #222222 !important;
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

# Logo and title (no black top bar)
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

    st.markdown(f"**Animal:** {animal_label}")
    st.markdown(f"**Breed:** {breed_name} ({confidence:.2%} confidence)")

    st.markdown('<div class="status-text">Generating breed description...</div>', unsafe_allow_html=True)

    description = get_breed_description(label.lower(), breed_name)
    st.markdown(f"**Breed Description:**\n\n{description}")
