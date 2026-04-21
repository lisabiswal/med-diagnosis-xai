import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import os

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Attach hooks
        self.hooks = [
            target_layer.register_forward_hook(self.save_activation),
            target_layer.register_full_backward_hook(self.save_gradient)
        ]
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
            
    def generate_heatmap(self, input_tensor, class_idx=None):
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
            
        # Backward pass
        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()
        
        # Calculate weights (global average pooling of gradients)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Calculate Grad-CAM
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-7)
        
        return cam.detach().cpu().numpy().squeeze(), class_idx

def save_gradcam_image(heatmap, original_image: Image.Image, save_path, alpha=0.4):
    """
    Overlays the heatmap on the original image and saves it.
    """
    # Resize heatmap to match original image
    heatmap_resized = cv2.resize(heatmap, (original_image.width, original_image.height))
    
    # Convert to 0-255 range and apply JET colormap
    heatmap_color = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_color, cv2.COLORMAP_JET)
    
    # Convert original image to BGR for OpenCV
    original_bgr = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
    
    # Overlay heatmap on original image
    overlay = cv2.addWeighted(original_bgr, 1.0, heatmap_color, alpha, 0)
    
    # Save image
    cv2.imwrite(save_path, overlay)
    return save_path
