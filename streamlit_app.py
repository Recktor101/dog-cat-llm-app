import streamlit as st
from PIL import Image
from model import predict_image
from description_model import get_breed_description

st.set_page_config(page_title="Dog and Cat Image Classifier", layout="centered")

# CSS to shrink the drag and drop area/button, make line white, adjust spacing and size
st.markdown(
    """
    <style>
    /* Shrink and widen the file uploader label/button */
    div[data-testid="fileUploaderDropzone"] > label {
        padding: 0.3rem 1rem !important;  /* more horizontal padding for width */
        font-size: 14px !important;       /* slightly bigger text */
        max-width: 180px;                 /* wider to fit label */
        white-space: nowrap;              /* keep label on one line */
        margin-left: 8px !important;     /* closer to the left text */
        cursor: pointer;
    }
    /* Make horizontal line white */
    hr {
        border-color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header with logo and title (title stays dark blue)
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

# Create two columns: label left, uploader right, reduce gap
col1, col2 = st.columns([4,1])

with col1:
    st.markdown(
        """
        <div style="display: flex; align-items: center; height: 100%;">
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
