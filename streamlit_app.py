import streamlit as st
from PIL import Image
from model import predict_image
from description_model import get_breed_description

st.set_page_config(page_title="Dog and Cat Image Classifier", layout="centered")

# CSS styles
st.markdown(
    """
    <style>
    /* Make the file uploader smaller in width and taller in height, vertical text label */
    div[data-testid="fileUploaderDropzone"] > label {
        width: 90px !important;
        height: 70px !important;
        padding: 0.4rem !important;
        font-size: 14px !important;
        writing-mode: vertical-rl;
        text-orientation: mixed;
        display: flex !important;
        justify-content: center;
        align-items: center;
        white-space: nowrap;
        cursor: pointer;
        background-color: #f0f0f0;
        border-radius: 8px;
        border: 1px solid #ddd;
        color: #444;
        user-select: none;
        margin-left: 8px !important;
    }

    /* Hide the default file input text to just show custom label */
    input[type="file"] {
        display: none !important;
    }

    /* White horizontal line */
    hr {
        border-color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header with logo and title (dark blue)
st.markdown(
    """
    <div style="text-align: center; margin-bottom: 5px;">
        <img src="https://raw.githubusercontent.com/Recktor101/dog-cat-llm-app/main/assets/llmatscale-logo.png" width="250">
        <div style="font-size: 10px; color: lightgray; margin-top: 4px;">
            LLM at Scale
        </div>
        <div style="font-size: 24px; font-weight: 700; color: #003366; margin-top: 8px;">
            Dog and Cat Image Classifier
        </div>
    </div>
    <hr style="margin-top: 10px; margin-bottom: 20px;">
    """,
    unsafe_allow_html=True,
)

# Two columns: label text left, vertical drag-and-drop file uploader right
col1, col2 = st.columns([4,1])

with col1:
    st.markdown(
        """
        <div style="display: flex; align-items: center; height: 70px;">
            <span style="font-size: 18px; font-weight: 500; color: white;">
                Upload an image of a dog or cat
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="uploader")

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
