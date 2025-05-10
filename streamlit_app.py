import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from transformers import pipeline

# Load image classifier
@st.cache_resource
def load_model():
    model = torch.hub.load("pytorch/vision", "resnet18", pretrained=True)
    model.eval()
    return model

# Load Hugging Face LLM
@st.cache_resource
def load_llm():
    return pipeline("text-generation", model="tiiuae/falcon-7b-instruct")

# Preprocess uploaded image
def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

# Map model output index to label
def get_label(index):
    if 281 <= index <= 285:
        return "cat"
    elif 151 <= index <= 268:
        return "dog"
    else:
        return "neither a dog nor a cat"

# Streamlit interface
st.title("🐶🐱 Dog or Cat Identifier + Description")

uploaded_file = st.file_uploader("Upload a dog or cat image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    model = load_model()
    input_tensor = preprocess(image)

    with torch.no_grad():
        output = model(input_tensor)
    pred_idx = torch.argmax(output).item()

    label = get_label(pred_idx)
    st.subheader(f"Prediction: **{label.upper()}**")

    if label in ["dog", "cat"]:
        llm = load_llm()
        prompt = f"Describe a {label}. Include care tips and personality traits."
        result = llm(prompt, max_new_tokens=100)[0]["generated_text"]
        st.subheader("AI Description:")
        st.write(result.strip())
    else:
        st.warning("This image doesn't appear to be a dog or a cat.")
