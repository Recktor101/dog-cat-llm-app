import torch
from torchvision import transforms
from PIL import Image

# Load breed labels once
with open('breed_labels.txt', 'r') as f:
    breed_labels = [line.strip() for line in f.readlines()]

# Load your pretrained model file (make sure model.pth is in your repo)
model = torch.load('model.pth', map_location='cpu')
model.eval()

# Image transform pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Normalize if your model expects it, example:
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225]),
])

def predict_breed(image_path):
    # Open and transform image
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0)  # add batch dim

    with torch.no_grad():
        outputs = model(input_tensor)
        preds = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    idx = preds.argmax()
    confidence = preds[idx]

    if idx >= len(breed_labels):
        raise ValueError(f"Prediction index {idx} out of range for breed labels")

    breed = breed_labels[idx]

    # More detailed description example (customize as you want)
    description = (
        f"The {breed.replace('_', ' ')} is a lovely breed known for its unique characteristics. "
        f"This prediction has a confidence of {confidence*100:.2f}%."
    )

    return breed, description, confidence
