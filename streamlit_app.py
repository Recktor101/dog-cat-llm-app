import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from transformers import pipeline

# Load ImageNet class labels
@st.cache_resource
def load_imagenet_labels():
    import json
    from urllib.request import urlopen
    labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    labels = urlopen(labels_url).read().decode("utf-8").splitlines()
    return labels

# Load ResNet18 model
@st.cache_resource
def load_image_model():
    model = torch.hub.load("pytorch/vision", "resnet18", pretrained=True)
    model.eval()
    return model

# Load Flan-T5 text generator
@st.cache_resource
def load_text_model():
    return pipeline("text2text-generation", model="google/flan-t5-base")

# Preprocess uploaded image
def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

# Classify as dog, cat, or neither based on label text
def classify_pet(label):
    label_lower = label.lower()
    if "cat" in label_lower:
        return "cat"
    elif any(dog_word in label_lower for dog_word in ["dog", "hound", "retriever", "terrier", "spaniel", "shepherd", "poodle", "bulldog"]):
        return "dog"
    else:
        return "neither"

# Streamlit UI
st.title("🐶🐱 Dog or Cat Classifier + Description Generator")

uploaded_file = st.file_uploader("Upload an image of a dog or cat", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    model = load_image_model()
    labels = load_imagenet_labels()
    input_tensor = preprocess(image)

    with torch.no_grad():
        output = model(input_tensor)
    pred_idx = torch.argmax(output).item()
    class_label = labels[pred_idx]
    confidence = torch.nn.functional.softmax(output[0], dim=0)[pred_idx].item()

    st.subheader(f"🧠 Predicted: **{class_label.title()}**")
    st.caption(f"Confidence: {confidence:.2f}")

    pet_type = classify_pet(class_label)

    if pet_type in ["dog", "cat"] and confidence >= 0.5:
        gen = load_text_model()
        prompt = f"Write an informative and friendly description of a {class_label}. Include common traits, behavior, and care tips."
        result = gen(prompt, max_new_tokens=200, temperature=0.7)[0]["generated_text"]

        st.subheader("📘 Description from Flan-T5:")
        st.write(result.strip())
    else:
        st.warning("⚠️ The model isn't confident that this is a dog or a cat. Please try another image.")
