import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# Define your CNN model architecture
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=120):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 256),  # Assuming input images 128x128
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Load breed labels
with open("breed_labels.txt", "r") as f:
    breed_labels = [line.strip() for line in f.readlines()]

# Load breed descriptions
with open("breed_descriptions.txt", "r") as f:
    breed_descriptions = {}
    for line in f:
        # Format: BreedName|Description
        if "|" in line:
            breed, desc = line.strip().split("|", 1)
            breed_descriptions[breed] = desc

# Load model weights (make sure model.pth is in your repo)
model = SimpleCNN(num_classes=len(breed_labels))
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model.eval()

# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # standard ImageNet normalization
                         std=[0.229, 0.224, 0.225]),
])

def predict_breed(image_path):
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, idx = torch.max(probs, 1)
        breed = breed_labels[idx.item()]
        description = breed_descriptions.get(breed, "Description not available.")
    return breed, description, confidence.item()
