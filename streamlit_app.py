import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from transformers import pipeline
import os

# Load image classification model
@st.cache_resource
def load_image_model():
    model = torch.hub.load("pytorch/vision", "resnet18", pretrained=True)
    model.eval()
    return model

# Load Hugging Face text generation pipeline
@st.cache_resource
def load_llm():
    hf_token = os.environ.get("HF_TOKEN")  # Secure Hugging Face token via environment variable
    if not hf_token:
        st.error("Hugging Face token not found. Make sure it's set as a secret.")
        return None
    return pipeline("text-generation", model="tiiuae/falcon-7b-instruct", token=hf_token)

# Preprocess image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

# Get label from ImageNet index
def get_label(index):
    if 281 <= index <= 285:
        return "cat"
    elif 151 <= index <= 268:
        return "dog"
    else:
        return "neither a dog nor a cat"

# Streamlit UI
st.title("🐶🐱 Dog or Cat Identifier + AI Description")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    model = load_image_model()
    input_tensor = preprocess_image(image)

    with torch.no_grad():
        output = model(input_tensor)
    pred_idx = torch.argmax(output).item()
    label = get_label(pred_idx)
    st.subheader(f"Prediction: **{label.upper()}**")

    if label in ["dog", "cat"]:
        llm = load_llm()
        if llm:
            prompt = f"Describe a {label}. Include care tips, characteristics, and personality."
            result = llm(prompt, max_new_tokens=100)[0]["generated_text"]
            st.subheader("🧠 AI Description:")
            st.write(result.strip())
        else:
            st.warning("Failed to load LLM.")
    else:
        st.warning("This does not look like a dog or a cat.")
