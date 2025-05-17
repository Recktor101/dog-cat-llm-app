import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import urllib.request

model = models.resnet50(pretrained=True)
model.eval()

url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels = urllib.request.urlopen(url).read().decode("utf-8").splitlines()

DOG_BREEDS = [
    "dog", "labrador", "poodle", "beagle", "boxer", "bulldog", "dalmatian",
    "doberman", "golden retriever", "german shepherd", "chihuahua", "pug",
    "shih tzu", "rottweiler", "corgi", "great dane", "husky", "jack russell",
    "maltese", "newfoundland", "papillon", "pekinese", "pomeranian", "pug",
    "saint bernard", "samoyed", "scottish terrier", "staffordshire bullterrier",
    "weimaraner", "whippet"
]

CAT_BREEDS = [
    "cat", "tabby", "siamese cat", "persian cat", "egyptian cat", "tiger cat",
    "lynx", "leopard", "cheetah", "jaguar", "lion", "tiger"
]

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
    predicted_label = labels[predicted_idx.item()].lower()

    # check if predicted label matches dog breeds
    if any(dog_breed in predicted_label for dog_breed in DOG_BREEDS):
        return "dog", predicted_label.title(), confidence.item()

    # check if predicted label matches cat breeds
    elif any(cat_breed in predicted_label for cat_breed in CAT_BREEDS):
        return "cat", predicted_label.title(), confidence.item()

    else:
        # Neither dog nor cat detected confidently
        return "unknown", predicted_label.title(), confidence.item()
