import streamlit as st
from PIL import Image
from model import predict_image
from description_model import get_breed_description

st.set_page_config(page_title="Dog and Cat Image Classifier", layout="centered")

# CSS for making the file uploader button smaller
st.markdown(
    """
    <style>
    /* Shrink the file uploader button */
    div[data-testid="fileUploaderDropzone"] > label {
        padding: 0.2rem 0.5rem !important;
        font-size: 12px !important;
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
        <div style="font-size: 24px; font-weight: 700; color: #003366; margin-top: 8px;">
            Dog and Cat Image Classifier
        </div>
    </div>
    <hr style="margin-top: 10px; margin-bottom: 20px; border-color: #003366;">
    """,
    unsafe_allow_html=True,
)

# Create two columns with some width ratio for uploader button and label
col1, col2, col3 = st.columns([1, 0.1, 3])

with col1:
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="uploader")

with col2:
    st.write("")  # small gap column

with col3:
    st.markdown(
        """
        <div style="display: flex; align-items: center; height: 100%;">
            <span style="font-size: 16px; font-weight: 500; color: #003366;">
                Upload an image of a dog or cat
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
