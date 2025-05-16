import torch
from PIL import Image
from torchvision import transforms
import timm

# Load model with 120 breed classes
model = timm.create_model('tf_efficientnet_b2_ns', pretrained=True, num_classes=120)
model.eval()

# Load breed labels
with open("breed_labels.txt", "r") as f:
    breed_labels = [line.strip() for line in f.readlines()]

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_dog_breed(image: Image.Image):
    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(img)
        pred_idx = logits.argmax(dim=1).item()
    return breed_labels[pred_idx]
