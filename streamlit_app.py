port streamlit as st
from PIL import Image
from model import predict_image
from description_model import get_breed_description

# Set page configuration
st.set_page_config(page_title="Dog and Cat Image Classifier", layout="centered")

# Display the centered logo, then centered small title, then horizontal line
st.markdown(
    """
    <div style="text-align: center; margin-bottom: 5px;">
        <img src="https://raw.githubusercontent.com/Recktor101/dog-cat-llm-app/main/assets/llmatscale-logo.png" width="250">
        <div style="font-size: 10px; color: gray; margin-top: 4px;">
            LLM at Scale
        </div>
        <div style="font-size: 14px; font-weight: 600; color: black; margin-top: 8px;">
            Dog and Cat Image Classifier
        </div>
    </div>
    <hr style="margin-top: 10px; margin-bottom: 20px;">
    """,
    unsafe_allow_html=True
)

# File uploader
uploaded_file = st.file_uploader("Upload an image of a dog or cat", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Add 'alien' text centered below image in smaller font
    st.markdown(
        """
        <div style="text-align: center; font-size: 12px; color: gray; margin-bottom: 15px;">
            alien
        </div>
        """,
        unsafe_allow_html=True
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
