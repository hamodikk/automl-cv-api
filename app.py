# app.py

import io
import torch
import mlflow.pytorch
from fastapi import FastAPI, File, UploadFile
from torchvision import transforms
from PIL import Image
from src import config
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="AutoML-CV Model API",
    description="Serve predictions from the best image classification model.",
    version="1.0.0"
)

# Load the model from MLflow
print("Loading model from MLflow...")
model = mlflow.pytorch.load_model("model")
model.eval()

# Define preprocessing
transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

MAX_UPLOAD_SIZE_MB = 5
MAX_UPLOAD_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

# Predict endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        if len(image_bytes) > MAX_UPLOAD_BYTES:
            return {"error": f"Image too large. Max size is {MAX_UPLOAD_SIZE_MB}MB."}
        
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probs, dim=0)
            label = config.CLASS_NAMES[predicted_idx.item()]
        return {
            "label": label,
            "confidence": round(confidence.item(), 4),
            "class_index": predicted_idx.item()
        }
    except Exception as e:
        return {"error": str(e)}
    
# Serve index.html on root route "/"
@app.get("/", response_class=HTMLResponse)
async def serve_home():
    return FileResponse("static/index.html")
    
# Serve HTML and static assets from /static
app.mount("/static", StaticFiles(directory="static", html=True), name="static")
