import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the breed classifier model (A specialized dog breed classifier from Hugging Face)
@st.cache_resource
def load_breed_classifier():
    model = models.resnet50(pretrained=True)
    model.eval()  # Set model to evaluation mode
    return model

# Load Flan-T5 model (for generating breed descriptions)
@st.cache_resource
def load_flan_t5_model():
    model_name = "google/flan-t5-small"
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model.eval()
    return model, tokenizer

# Image transformation function for preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image for model input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Standard ImageNet normalization
    ])
    return transform(image).unsqueeze(0)

# Function to classify breed (Dog Breed Classifier)
def classify_breed(model, image):
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(input_tensor)
    pred_idx = torch.argmax(output).item()

    # Example breed mappings
    breed_classes = {243: "Chihuahua", 244: "Japanese Chin", 245: "Maltese dog", 246: "Pekingese", 
                     247: "Shih-Tzu", 248: "Yorkshire Terrier", 151: "Labrador Retriever", 152: "Golden Retriever"}
    return breed_classes.get(pred_idx, "Unknown Breed")

# Function to generate a breed-specific description using Flan-T5 model
def generate_breed_description(model, tokenizer, breed_name):
    prompt = f"Please provide a detailed description of a {breed_name}. Include its physical traits, personality, care tips, and history."
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(inputs['input_ids'], max_length=150, num_return_sequences=1)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Streamlit UI
st.title("🐶🐱 Dog or Cat Classifier + Breed Description Generator")

uploaded_file = st.file_uploader("Upload an image of a dog or cat", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load the breed classification model
    breed_classifier = load_breed_classifier()

    # Predict breed of the uploaded image
    breed_name = classify_breed(breed_classifier, image)

    # Load Flan-T5 model for text generation
    flan_t5_model, flan_t5_tokenizer = load_flan_t5_model()

    # Generate breed-specific description using Flan-T5
    description = generate_breed_description(flan_t5_model, flan_t5_tokenizer, breed_name)

    # Display breed description
    st.subheader(f"Prediction: **{breed_name.upper()}**")
    st.subheader("🧠 AI-Generated Breed Description:")
    st.write(description.strip())

else:
    st.warning("Please upload an image of a dog or cat.")
