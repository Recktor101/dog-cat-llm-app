import torch
from torchvision import transforms
from PIL import Image

# Load breed labels globally once
with open('breed_labels.txt', 'r') as f:
    breed_labels = [line.strip() for line in f.readlines()]

# Load your model (make sure the path is correct)
model = torch.load('model.pth', map_location='cpu')
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # normalize if needed
])

def predict_breed(image_path):
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(input_tensor)
        preds = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    idx = preds.argmax()

    # Safety check
    if idx >= len(breed_labels):
        raise ValueError(f"Predicted index {idx} exceeds number of labels {len(breed_labels)}")

    breed = breed_labels[idx]
    confidence = preds[idx]

    # Simple description example, replace with better later
    description = f"{breed} is a wonderful breed."

    return breed, description, confidence
