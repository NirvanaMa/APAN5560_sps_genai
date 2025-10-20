from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import torchvision.transforms as transforms
from PIL import Image
from helper_lib.model import get_model

app = FastAPI(title="CIFAR10 Classifier API")

# CIFAR-10 classes
CLASSES = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# Load model
device = "cpu"
model = get_model("CNN64", num_classes=10).to(device)
model.load_state_dict(torch.load("checkpoints/cnn64_cifar10.pth", map_location=device))
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010])
])

@app.get("/")
def root():
    return {"message": "CIFAR10 FastAPI server is running 🚀"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img = Image.open(file.file).convert("RGB")
        img_t = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_t)
            probs = torch.softmax(outputs, dim=1)
            pred_idx = probs.argmax(dim=1).item()
            confidence = probs[0, pred_idx].item()

        return {"prediction": CLASSES[pred_idx],
                "confidence": round(confidence, 4)}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)