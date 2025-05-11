import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
from transformers import T5ForConditionalGeneration, T5Tokenizer, BlipProcessor, BlipForConditionalGeneration

# Load Flan-T5 model (for breed descriptions)
@st.cache_resource
def load_flan_t5_model():
    model_name = "google/flan-t5-small"  # Hugging Face model name for Flan-T5
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.eval()  # Set the model to evaluation mode
    return model, tokenizer

# Load BLIP model (for image captioning)
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# Load the breed classifier model (using ResNet50 for dog breed classification)
@st.cache_resource
def load_breed_classifier():
    model = models.resnet50(pretrained=True)
    model.eval()  # Set model to evaluation mode
    return model

# Image transformation function for preprocessing (ResNet50 input)
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

    # You need to load a proper breed-to-index mapping dictionary for dog breeds
    breed_classes = {243: "Chihuahua", 244: "Japanese Chin", 245: "Maltese dog", 246: "Pekingese", 
                     247: "Shih-Tzu", 248: "Yorkshire Terrier", 151: "Labrador Retriever", 152: "Golden Retriever"}  # Example breed mappings
    return breed_classes.get(pred_idx, "Unknown Breed")

# Function to generate breed-specific description using Flan-T5
def generate_breed_description(model, tokenizer, breed_name):
    prompt = f"Describe a {breed_name} dog. Include physical traits, personality, care tips, and history."
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(inputs['input_ids'], max_length=150, num_return_sequences=1)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Function to generate an image caption using BLIP
def generate_image_caption(processor, model, image):
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# Streamlit UI
st.title("🐶🐱 Dog or Cat Classifier + Breed Description Generator")

# File uploader to upload the image
uploaded_file = st.file_uploader("Upload an image of a dog or cat", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Open and display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load the breed classifier model
    breed_classifier = load_breed_classifier()

    # Predict breed of the uploaded image
    breed_name = classify_breed(breed_classifier, image)

    # Load Flan-T5 model for breed description generation
    flan_t5_model, flan_t5_tokenizer = load_flan_t5_model()

    # Generate breed-specific description using Flan-T5
    description = generate_breed_description(flan_t5_model, flan_t5_tokenizer, breed_name)

    # Display the breed classification and description
    st.subheader(f"Prediction: **{breed_name.upper()}**")
    st.subheader("🧠 AI-Generated Breed Description:")
    st.write(description.strip())

    # Load BLIP model for image captioning
    blip_processor, blip_model = load_blip_model()

    # Generate a caption for the uploaded image
    image_caption = generate_image_caption(blip_processor, blip_model, image)

    # Display the AI-generated image caption
    st.subheader("🖼️ AI-Generated Image Caption:")
    st.write(image_caption.strip())

else:
    st.warning("Please upload an image of a dog or cat.")
