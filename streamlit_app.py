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

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background-color: black;
        color: white;
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
    image = resize_with_aspect_ratio(image, max_size=300)

    # Display image centered
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

    animal, breed_name, confidence = predict_image(image)

    # Make animal lowercase for display, e.g. "dog" or "cat"
    animal_display = animal.lower() if animal else "unknown"

    st.markdown(f"**Animal:** {animal_display}")
    st.markdown(f"**Breed:** {breed_name} ({confidence:.2%} confidence)")

    if animal.lower() == "dog":
        st.markdown('<div class="status-text">Generating breed description...</div>', unsafe_allow_html=True)
        description = get_breed_description(animal, breed_name)
        st.markdown(f"**Breed Description:**\n\n{description}")
    elif animal.lower() == "cat":
        st.markdown('<div class="status-text">Generating cat breed description (coming soon)...</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-text">Breed description not available for unknown animals.</div>', unsafe_allow_html=True)
