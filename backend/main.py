import os
import uuid
import time
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
import io
from model import predict, model, transform, DEVICE
from xai import GradCAM, save_gradcam_image

app = FastAPI(title="Medical XAI API")

# Initialize GradCAM
# target_layer = model.layer4 for ResNet18
gcam = GradCAM(model, model.layer4)

# To also run without cuda
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Running Backend on: {DEVICE} ---")

# heatmap path
os.makedirs("static/heatmaps", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# API Response Schema 
class AnalysisResponse(BaseModel):
    prediction: str
    confidence: float
    processing_time: float
    heatmap_url: str | None = None

# Mock Inference Engine
# def run_mock_inference(image: Image.Image):
#     time.sleep(2) 
    
#     # Simulating a medical result
#     prediction = "Abnormal (Potential Fracture Detected)"
#     confidence = 0.9423
    
#     # Placeholder for the XAI heatmap URL
#     mock_heatmap_path = "/static/heatmaps/sample_heatmap.png" 
    
#     return prediction, confidence, mock_heatmap_path

# The Inference Endpoint
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid image type.")

    start_time = time.time()
    
    # Generate a Unique Request ID
    request_id = str(uuid.uuid4())
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Simulate the XAI Heatmap creation with the UUID
        heatmap_filename = f"heatmap_{request_id}.png"
        heatmap_rel_path = f"/static/heatmaps/{heatmap_filename}"
        
       
        # Real model call
        label, score = predict(image)

        # Generate Grad-CAM Heatmap
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        heatmap, _ = gcam.generate_heatmap(input_tensor)
        
        # Save Heatmap Overlay
        heatmap_full_path = os.path.join("static", "heatmaps", heatmap_filename)
        save_gradcam_image(heatmap, image, heatmap_full_path)

        end_time = time.time()
        
        return AnalysisResponse(
            prediction=label,
            confidence=score,
            processing_time=round(end_time - start_time, 2),
            heatmap_url=heatmap_rel_path
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    
# Health Check
@app.get("/")
def read_root():
    return {"message": "Medical XAI Backend is Live", "device": str(DEVICE)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)