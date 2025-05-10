import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from transformers import pipeline

# Load the image classifier (ResNet18)
@st.cache_resource
def load_image_model():
    model = torch.hub.load("pytorch/vision", "resnet18", pretrained=True)
    model.eval()
    return model

# Load Flan-T5 from Hugging Face for text generation
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

# Identify if it's a dog, cat, or neither based on ImageNet class index
def get_label(index):
    if 281 <= index <= 285:
        return "cat"
    elif 151 <= index <= 268:
        return "dog"
    else:
        return "neither a dog nor a cat"

# Refined breed-specific description prompt
def generate_breed_description(animal_type, breed_name):
    prompt = f"Provide a detailed description of a {animal_type} of the {breed_name} breed. Include characteristics, appearance, common personality traits, and care tips specific to the {breed_name}. Make the tone friendly and informative."
    return prompt

# Streamlit UI
st.title("🐶🐱 Dog or Cat Classifier + Description Generator")

uploaded_file = st.file_uploader("Upload an image of a dog or cat", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    model = load_image_model()
    input_tensor = preprocess(image)

    with torch.no_grad():
        output = model(input_tensor)
    pred_idx = torch.argmax(output).item()

    label = get_label(pred_idx)
    st.subheader(f"Prediction: **{label.upper()}**")

    if label in ["dog", "cat"]:
        # Define breeds for the specific animals (this would need more customization based on real classification)
        breed_name = "Labrador" if label == "dog" else "Siamese"  # Customize this based on the breed identified from the model
        
        # Generate prompt
        prompt = generate_breed_description(label, breed_name)
        
        # Generate breed description
        gen = load_text_model()
        result = gen(prompt, max_new_tokens=200, temperature=0.7)[0]["generated_text"]

        st.subheader("🧠 AI-Generated Breed Description:")
        st.write(result.strip())
    else:
        st.warning("The image does not appear to be a dog or cat.")
