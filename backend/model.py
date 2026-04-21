import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# build + load trained model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model.eval().to(DEVICE)

# same preprocessing used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def predict(image: Image.Image):
    img = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = F.softmax(model(img), dim=1)
    
    conf, cls = torch.max(probs, 1)

    label = "Normal" if cls.item() == 0 else "Abnormal"
    return label, conf.item()