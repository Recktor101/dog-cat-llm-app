import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the FLAN-T5 model and tokenizer
def load_flan_t5_model():
    model_name = "google/flan-t5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return model, tokenizer

# Load ResNet model for image classification
def load_resnet_model():
    model = torch.hub.load("pytorch/vision", "resnet18", pretrained=True)
    model.eval()
    return model

# Preprocess image for ResNet model
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

# Classify the image as dog, cat, or other
def get_label(index):
    if 281 <= index <= 285:
        return "cat"
    elif 151 <= index <= 268:
        return "dog"
    else:
        return "neither a dog nor a cat"

# Streamlit UI setup
st.title("🐶🐱 Dog or Cat Identifier + Description Generator")

uploaded_file = st.file_uploader("Upload an image of a dog or cat", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load the ResNet model and classify the image
    resnet_model = load_resnet_model()
    input_tensor = preprocess_image(image)

    with torch.no_grad():
        output = resnet_model(input_tensor)
    pred_idx = torch.argmax(output).item()

    # Get the prediction label
    label = get_label(pred_idx)
    st.subheader(f"Prediction: **{label.upper()}**")

    # If label is Dog or Cat, get a description using FLAN-T5
    if label in ["dog", "cat"]:
        model, tokenizer = load_flan_t5_model()
        prompt = f"Describe a {label}. Include personality and care tips."
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        summary_ids = model.generate(inputs['input_ids'], max_length=100, num_beams=5, early_stopping=True)
        description = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        st.subheader("🧠 AI-Generated Description:")
        st.write(description.strip())
    else:
        st.warning("The image does not appear to be a dog or cat.")
