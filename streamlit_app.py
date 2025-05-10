import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from transformers import pipeline

# Load a simple image classifier (e.g., a fine-tuned ResNet)
@st.cache_resource
def load_model():
    model = torch.hub.load("pytorch/vision", "resnet18", pretrained=True)
    model.eval()
    return model

# Load LLaMA model via Hugging Face API for text generation
@st.cache_resource
def load_llama():
    try:
        st.write("Loading LLaMA model...")  # Debug log
        llama_model = pipeline("text-generation", model="meta-llama/LLaMA-7B-hf")
        st.write("LLaMA model loaded successfully!")  # Debug log
        return llama_model
    except Exception as e:
        st.error(f"Error loading LLaMA model: {e}")
        return None

# Image transform
def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

# Label mapping (ImageNet dog/cat classes)
def get_label(index):
    if 281 <= index <= 285:
        return "cat"
    elif 151 <= index <= 268:
        return "dog"
    else:
        return "neither a dog nor a cat"

# Streamlit UI
st.title("🐶🐱 Dog or Cat Identifier + LLaMA Description")

uploaded_file = st.file_uploader("Upload an image of a dog or cat", type=["jpg", "jpeg", "png"])

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
        try:
            gen = load_llama()  # Load LLaMA model
            if gen is not None:
                prompt = f"Describe a {label}. Include care tips and personality traits."
                result = gen(prompt, max_length=100)[0]["generated_text"]
                st.subheader("🧠 Description:")
                st.write(result.strip())  # Display the response
                print(result)  # Print response in logs for debugging
            else:
                st.error("Failed to load LLaMA model")
        except Exception as e:
            st.error(f"Error generating description: {str(e)}")
    else:
        st.warning("The image does not appear to be a dog or cat.")
