import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
from transformers import pipeline

# Load the breed classifier model (ResNet50 trained on the ImageNet dataset for general classification)
@st.cache_resource
def load_breed_classifier():
    # Load the ResNet50 model
    model = models.resnet50(pretrained=True)
    model.eval()
    return model

# Load the text generation model (Flan-T5 for breed description)
@st.cache_resource
def load_text_model():
    return pipeline("text2text-generation", model="google/flan-t5-base")

# Image transformation function for preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image for model input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Standard ImageNet normalization
    ])
    return transform(image).unsqueeze(0)

# Function to classify breed (ResNet50)
def classify_breed(model, image):
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(input_tensor)
    pred_idx = torch.argmax(output).item()
    
    # Here we need the label mapping to the actual breed names. We'll use ImageNet classes for now.
    # Mapping to dog breeds - This mapping can be replaced with more accurate breed names if necessary.
    breed_classes = {243: "Chihuahua", 244: "Japanese Chin", 245: "Maltese dog", 246: "Pekingese", 
                     247: "Shih-Tzu", 248: "Yorkshire Terrier", 151: "Labrador Retriever", 152: "Golden Retriever"}  # Example breed mappings
    return breed_classes.get(pred_idx, "Unknown Breed")

# Function to generate a breed-specific description
def generate_breed_description(breed_name):
    prompt = f"Please provide a detailed description of a {breed_name}. Include its physical traits, personality, care tips, and history."
    return prompt

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

    # Generate breed-specific description
    prompt = generate_breed_description(breed_name)

    # Load text generation model (Flan-T5 or another model)
    text_model = load_text_model()
    result = text_model(prompt, max_new_tokens=200, temperature=0.7)[0]["generated_text"]

    # Display breed description
    st.subheader(f"Prediction: **{breed_name.upper()}**")
    st.subheader("🧠 AI-Generated Breed Description:")
    st.write(result.strip())

else:
    st.warning("Please upload an image of a dog or cat.")
