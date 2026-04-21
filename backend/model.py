import torch
from torchvision import models, transforms
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=True)
model.eval()
model.to(DEVICE)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict(image):
    img = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img)
        probs = F.softmax(outputs, dim=1)

    confidence, predicted = torch.max(probs, 1)

    # simple mapping
    label = "Abnormal" if confidence.item() > 0.5 else "Normal"

    return label, confidence.item()