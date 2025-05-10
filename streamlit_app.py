mport streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from transformers import pipeline

# Load the pre-trained breed classification model from Hugging Face
@st.cache_resource
def load_breed_classifier():
    # Use a pre-trained model for dog breeds classification
    model = torch.hub.load("pytorch/vision", "resnet50", pretrained=True)
    model.eval()
    return model

# Load the text generation model (Flan-T5 or another model for breed description)
@st.cache_resource
def load_text_model():
    return pipeline("text2text-generation", model="google/flan-t5-base")

# Image transformation function for preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image for model input
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

# Function to classify dog breed from image (using a pre-trained model)
def classify_breed(model, image):
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(input_tensor)
    pred_idx = torch.argmax(output).item()
    
    # This function needs to map the predicted index to breed names, 
    # which can be done using a class label mapping. Here, we just return a placeholder breed for now.
    breed_names = ["Golden Retriever", "Labrador Retriever", "Bulldog", "Beagle"]  # Add all breed names here
    return breed_names[pred_idx % len(breed_names)]  # Return a breed from a placeholder list

# Generate a breed description from the model
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
