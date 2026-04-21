import torch
from model import model, transform, DEVICE, predict
from xai import GradCAM, save_gradcam_image
from explain import generate_explanation
from PIL import Image
import os

def test_gradcam():
    print(f"Testing Grad-CAM on {DEVICE}...")
    
    # Load a sample image
    img_path = "abnormal.jpg"
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found.")
        return

    image = Image.open(img_path).convert("RGB")
    
    # Initialize GradCAM
    gcam = GradCAM(model, model.layer4)
    
    # Get prediction
    label, score = predict(image)
    print(f"Prediction: {label} ({score:.4f})")
    
    # Generate Heatmap
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    heatmap, class_idx = gcam.generate_heatmap(input_tensor)
    print(f"Generated heatmap for class {class_idx}")
    
    # Save Heatmap
    os.makedirs("static/heatmaps", exist_ok=True)
    save_path = "static/heatmaps/test_verification.png"
    save_gradcam_image(heatmap, image, save_path)
    print(f"Heatmap saved to {save_path}")
    
    # Generate Explanation
    explanation = generate_explanation(heatmap, label, score)
    print(f"Explanation: {explanation}")
    
    # Clean up hooks
    gcam.remove_hooks()
    print("Verification complete.")

if __name__ == "__main__":
    test_gradcam()
