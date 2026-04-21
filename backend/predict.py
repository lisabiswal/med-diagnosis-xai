from PIL import Image
from model import predict

if __name__ == "__main__":
    image_path = "normal.jpg"  

    image = Image.open(image_path).convert("RGB")
    label, confidence = predict(image)

    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.4f}")