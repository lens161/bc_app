import torch 
import torch.nn as nn
import uvicorn
import os

from fastapi import FastAPI, UploadFile, HTTPException, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
from pydantic import BaseModel
from torchvision.models import resnet18, ResNet18_Weights
from typing import *
from request import Request

def load_model(model_path):
    device = torch.device("cpu")
    model = resnet18(weights = ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    
    number_of_last_input = model.fc.in_features
    model.fc = nn.Linear(number_of_last_input,1)

    model.load_state_dict(torch.load(model_path, map_location=device))

    model.to(device)
    model.eval()
    return model

model = load_model(f"models/archive.pt")

app = FastAPI(debug=True)

requests: List[str] = [] # save requests in memory while program is running. no idea why, its just there...
# TODO: add get function to retrieve list of requests issued so far including their results. dont know why I need this. could be useful later...

memory_db = {"requests": requests}

origins = [
    "http://loacalhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if not os.path.exists("images/"):
    os.mkdir("images")


@app.get("/predict") # show list of requests
async def show_requests():
    return memory_db

@app.post("/predict")
async def predict(name: str = Form(...), img: UploadFile = File(...)):

    if img.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid image: wrong file type")

    image_file = await img.read()

    try:
        image = Image.open(BytesIO(image_file))
        image = image.convert("RGB")
    except Exception as e:
        print(f"image open error {e}")
        raise HTTPException(status_code=400, detail="invalid image, cant open")

    path = f"images/{img.filename}"
    with open(path, "wb") as f:
        f.write(image_file)

    request = Request(model, path)
    result = request.forward()
    memory_db["requests"].append(request.__str__())

    return JSONResponse({
        "name": name, 
        "filename": img.filename,
        "result": result
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)