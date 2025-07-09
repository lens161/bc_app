import torch 
import torch.nn as nn
import uvicorn

from fastapi import FastAPI, UploadFile, HTTPException, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
from pydantic import BaseModel
from torchvision import models
from typing import List

from request import Request

def load_model(model_path):
    device = torch.device("cpu")
    model = models.resnet18(pretrained =True)
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

origins = [
    "http://loacalhost:3000"
]

requests: List[Request] = [] # save requests in memory while program is running. no idea why, its just there...
# TODO: add get function to retrieve list of requests issued so far including their results. dont know why I need this. could be useful later...

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict/")
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
    requests.append(request)

    return JSONResponse({
        "name": name, 
        "filename": img.filename,
        "result": result
    })