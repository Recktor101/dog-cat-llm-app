import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# Load breed labels
with open("breed_labels.txt", "r") as f:
    breed_labels = [line.strip() for line in f.readlines()]

# Load pretrained model
model = models.resnet50(pretrained=True)
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

def predict_breed(image: Image.Image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        top_prob, top_idx = torch.max(probs, dim=0)

    breed = breed_labels[top_idx % len(breed_labels)]
    confidence = top_prob.item()
    return breed, confidence
