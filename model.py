import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import urllib.request

# Load pretrained ResNet50 model
model = models.resnet50(pretrained=True)
model.eval()

# Download actual ImageNet class labels
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels = urllib.request.urlopen(url).read().decode("utf-8").splitlines()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_image(image: Image.Image):
    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)

    confidence, predicted_idx = torch.max(probs, 0)
    predicted_label = labels[predicted_idx.item()]

    if "dog" in predicted_label.lower():
        return "DOG", predicted_label, confidence.item()
    else:
        return "CAT or UNKNOWN", predicted_label, confidence.item()
