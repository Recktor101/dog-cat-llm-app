import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# Load pretrained ResNet50 model
model = models.resnet50(pretrained=True)
model.eval()

# Breed labels for example purposes (you can expand this)
breed_labels = ['Beagle', 'Chihuahua', 'Doberman', 'French Bulldog', 'Golden Retriever', 'Maltese', 'Pug', 'Shih-Tzu']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_image(image: Image.Image):
    img = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(img)
    _, predicted = torch.max(outputs, 1)
    # Fake logic: just return breed for example
    breed = breed_labels[predicted.item() % len(breed_labels)]
    return "DOG", breed  # Always assume it's a dog for simplicity
