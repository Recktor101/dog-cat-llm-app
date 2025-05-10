import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import os

# Hugging Face token setup (REPLACE with your actual token)
HUGGINGFACE_TOKEN = "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"  # Don't include < or >

# Load image classification model (ResNet18)
@st.cache_resource
def load_classifier():
    model = torch.hub.load("pytorch/vision", "resnet18", pretrained=True)
    model.eval()
    return model

# Load LLaMA text generation model from Hugging Face
@st.cache_resource
def load_llama():
    try:
        model_name = "meta-llama/Llama-2-7b-hf"
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_auth_token=HUGGINGFACE_TOKEN
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_auth_token=HUGGINGFACE_TOKEN
        )
        gen_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
        return gen_pipeline
    except Exception as e:
        st.error(f"Error loading LLaMA model: {e}")
        return None

# Image preprocessing
def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

# ImageNet label mapping
def get_label(index):
    if 281 <= index <= 285:
        return "cat"
    elif 151 <= index <= 268:
        return "dog"
    else:
        return "neither a dog nor a cat"

# Streamlit UI
st.title("🐶🐱 Dog or Cat Identifier + LLaMA Breed Description")

uploaded_file = st.file_uploader("Upload an image of a dog or cat", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    classifier = load_classifier()
    input_tensor = preprocess(image)

    with torch.no_grad():
        output = classifier(input_tensor)
    pred_idx = torch.argmax(output).item()

    label = get_label(pred_idx)
    st.subheader(f"Prediction: **{label.upper()}**")

    if label in ["dog", "cat"]:
        gen = load_llama()
        if gen:
            prompt = (
                f"Please write a short, accurate, and informative description of a {label}. "
                f"Include specific care tips and common personality traits. "
                f"Do not confuse it with any other animal."
            )
            result = gen(prompt, max_new_tokens=100)[0]["generated_text"]
            st.subheader("🧠 AI-Generated Description:")
            st.write(result.strip())
    else:
        st.warning("The image does not appear to be a dog or cat.")
