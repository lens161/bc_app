import torch 
import torch.nn as nn
import uvicorn

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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

app = FastAPI(debug=True)

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
