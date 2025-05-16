import torch
from torchvision import transforms, models
from PIL import Image

# Load breed labels
with open("breed_labels.txt", "r") as f:
    breed_labels = [line.strip() for line in f.readlines()]

# Load pretrained model (replace with your actual model)
model = models.resnet50(pretrained=True)
model.eval()

# Preprocessing transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Some sample descriptions for popular breeds
breed_descriptions = {
    "Pug": "Pugs are charming, affectionate, and sturdy dogs with distinctive wrinkled faces and curled tails.",
    "Golden_retriever": "Golden Retrievers are friendly, intelligent, and devoted family dogs known for their dense golden coat.",
    "Chihuahua": "Chihuahuas are tiny but bold dogs, famous for their lively personality and loyalty.",
    "Siamese_cat": "Siamese cats are social, vocal, and striking with their blue almond-shaped eyes and short coat.",
    # add more as needed
}

def predict_breed(image_path):
    image = Image.open(image_path).convert("RGB")
    img_t = transform(image)
    batch_t = torch.unsqueeze(img_t, 0)

    with torch.no_grad():
        out = model(batch_t)
        probabilities = torch.nn.functional.softmax(out[0], dim=0)

    confidence, idx = torch.max(probabilities, 0)
    breed = breed_labels[idx]

    description = breed_descriptions.get(breed, f"{breed} is a wonderful breed loved by many.")
    confidence_percent = confidence.item() * 100

    return breed, description, confidence_percent
