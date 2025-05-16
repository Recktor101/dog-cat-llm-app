from model import predict_image
from description_model import get_breed_description

# ... previous code ...

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        st.write("🔍 Classifying the image...")
        label, breed, confidence = predict_image(image)

        st.markdown(f"### Prediction: **{label}**")
        st.markdown(f"### Breed: **{breed}**")
        st.markdown(f"### Confidence: **{confidence:.2%}**")

        if label == "DOG":
            st.write("📖 Generating breed description...")
            description = get_breed_description(breed)
            st.markdown(f"**Breed Description:**\n\n{description}")
