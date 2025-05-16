import streamlit as st
from PIL import Image
import tensorflow as tf
import torch
from torchvision import models, transforms
from transformers import T5ForConditionalGeneration, T5Tokenizer
import os
import numpy as np

# Load Dog vs Cat CNN model
cnn_model = tf.keras.models.load_model("utils/dog_cat_classifier.h5")

# Load FLAN-T5 model
t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

# Load ResNet50 for breed classification
resnet_model = models.resnet50(pretrained=True)
resnet_model.eval()

# Load ImageNet class names for breeds
with open("utils/breed_labels.txt") as f:
    breed_labels = [line.strip() for line in f.readlines()]

# Transform input for ResNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# --- Helper Functions ---

def predict_dog_or_cat(image):
    img = image.resize((128, 128))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    pred = cnn_model.predict(img_array)[0][0]
    return "DOG" if pred > 0.5 else "CAT"

def predict_breed(image):
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = resnet_model(img_tensor)
    _, predicted_idx = outputs.max(1)
    return breed_labels[predicted_idx.item()]

def generate_breed_description(breed_name):
    prompt = f"Describe the breed {breed_name} in 1-2 sentences."
    input_ids = t5_tokenizer(prompt, return_tensors="pt").input_ids
    outputs = t5_model.generate(input_ids, max_new_tokens=50)
    return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- Streamlit App ---

st.set_page_config(page_title="🐶🐱 Dog-Cat Breed Identifier")
st.title("🐾 Dog vs Cat Image Classifier + Breed Describer")

uploaded_file = st.file_uploader("Upload an image of a dog or cat", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Classifying..."):
        animal = predict_dog_or_cat(image)
        st.subheader(f"Prediction: {animal}")

        if animal == "DOG":
            breed = predict_breed(image)
            st.markdown(f"**Predicted Breed:** {breed}")
            description = generate_breed_description(breed)
            st.markdown(f"📖 **Breed Description:**\n\n{description}")
        else:
            st.markdown("🐱 It's a cat! No breed description needed.")
