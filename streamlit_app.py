import streamlit as st
from PIL import Image
from model import predict_image
from description_model import get_breed_description
import io
import base64

# A Function that resizes an uploaded image and keeping ratio
def resize_with_aspect_ratio(image, max_size=300):
    w, h = image.size
    if w > h:
        new_w = max_size
        new_h = int(h * max_size / w)
    else:
        new_h = max_size
        new_w = int(w * max_size / h)
    return image.resize((new_w, new_h))

# Page Set up With title
st.set_page_config(page_title="Dog and Cat Image Classifier", layout="centered")

#Full CSS for styling the uploading textbox
st.markdown(
    """
    <style>
    .top-black-bar {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 20px;
        background-color: black;
        z-index: 9999;
    }
    .main > div:first-child {
        padding-top: 20px;
    }

    /* Label above uploader */
    .custom-upload-label {
        font-size: 13px;
        color: black;
        font-weight: 500;
        margin-bottom: 6px;
        text-align: center;
    }

    /* Full black drag and drop uploader */
    div[data-testid="stFileUploader"] > div:first-child {
        background-color: black !important;
        color: white !important;
        border: 2px dashed white !important;
        border-radius: 10px;
        padding: 20px;
        cursor: pointer;
        text-align: center;
    }
    div[data-testid="stFileUploader"] > div:first-child:hover {
        background-color: #222222 !important;
    }
    div[data-testid="stFileUploader"] svg {
        color: white !important;
        fill: white !important;
    }
    div[data-testid="stFileUploader"] label {
        color: white !important;
        font-weight: 600;
    }
    </style>
    <div class="top-black-bar"></div>
    """,
    unsafe_allow_html=True,
)

#LLM At Scale Logo
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

#Label saying Upload a image of a Dog or Cat
st.markdown('<div class="custom-upload-label">Upload an image of a Dog or Cat</div>', unsafe_allow_html=True)

# # file upload color
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

# Once the button is clicked
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image = resize_with_aspect_ratio(image, max_size=300)

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    encoded_img = base64.b64encode(buf.getvalue()).decode()

    # Show upload image on screen
    st.markdown(
        f"""
        <div style="text-align: center;">
            <img src="data:image/png;base64,{encoded_img}" style="margin: auto;" />
        </div>
        """,
        unsafe_allow_html=True,
    )
    # SHow upload image on screen
    st.markdown('<div style="text-align:center; font-style:italic; color:#444;">Classifying the image...</div>', unsafe_allow_html=True)

    # Gives predictions and labels
    label, breed_name, confidence = predict_image(image)
    st.markdown(f"**Animal:** {label.capitalize()}")
    st.markdown(f"**Breed:** {breed_name} ({confidence:.2%} confidence)")

    st.markdown('<div style="text-align:center; font-style:italic; color:#444;">Generating breed description...</div>', unsafe_allow_html=True)

    description = get_breed_description(label.lower(), breed_name)
    st.markdown(f"**Breed Description:**\n\n{description}")
