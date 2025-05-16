import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from transformers import pipeline
import urllib.request
import json

# Load ImageNet labels
@st.cache_resource
def load_labels():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    labels = urllib.request.urlopen(url).read().decode("utf-8").splitlines()
    return labels

# Load the image classifier (ResNet18)
@st.cache_resource
def load_image_model():
    model = torch.hub.load("pytorch/vision", "resnet18", pretrained=True)
    model.eval()
    return model

# Load Flan-T5 from Hugging Face
@st.cache_resource
def load_text_model():
    return pipeline("text2text-generation", model="google/flan-t5-base")

# Preprocess the uploaded image
def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

# Check if the label is a dog or cat (based on ImageNet ranges)
def get_animal_type(index):
    if 281 <= index <= 285:
        return "cat"
    elif 151 <= index <= 268:
        return "dog"
    else:
        return "neither"

# UI
st.title("🐶🐱 Dog or Cat Classifier + Breed Description")

uploaded_file = st.file_uploader("Upload an image of a dog or cat", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    model = load_image_model()
    labels = load_labels()
    input_tensor = preprocess(image)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()

    predicted_label = labels[pred_idx]
    animal_type = get_animal_type(pred_idx)

    st.subheader(f"Prediction: **{predicted_label.upper()}** ({confidence*100:.2f}% confidence)")

    if animal_type in ["dog", "cat"] and confidence > 0.5:
        gen = load_text_model()

        # Prompt for better output with structure + repetition control
        prompt = f"""
        Write a friendly and informative guide about the {predicted_label}.
        Include:
        - A brief description of the breed
        - Common behavior traits
        - How to care for them (exercise, grooming, etc.)
        Avoid repeating the same phrases. Keep it natural and engaging.
        """

        result = gen(
            prompt,
            max_new_tokens=200,
            temperature=0.7,
            repetition_penalty=1.2,
        )[0]["generated_text"]

        st.subheader("📖 Breed Description:")
        st.write(result.strip())

    else:
        st.warning("This doesn't appear to be a dog or cat, or the confidence was too low.")
