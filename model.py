import torch
import torchvision.transforms as transforms
from PIL import Image

# Load breed labels
with open('breed_labels.txt', 'r') as f:
    breed_labels = [line.strip() for line in f.readlines()]

# Load the trained model
model = torch.load('model.pth', map_location='cpu')
model.eval()

# Image transforms (adjust to your model training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_breed(image_path):
    image = Image.open(image_path).convert('RGB')
    img_t = transform(image)
    batch_t = torch.unsqueeze(img_t, 0)

    with torch.no_grad():
        out = model(batch_t)
        probs = torch.nn.functional.softmax(out, dim=1)
        confidence, idx = torch.max(probs, 1)

    breed = breed_labels[idx.item()]
    confidence_percent = confidence.item() * 100

    # Basic description example
    description = f"The image is predicted as a {breed.replace('_', ' ')} with confidence {confidence_percent:.2f}%."

    return breed, description, confidence_percent
