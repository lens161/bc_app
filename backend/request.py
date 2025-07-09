import datetime
import numpy as np 
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.io import read_image

class Request:
    def __init__(self, model, image_path):
        self.timestamp = datetime.datetime.now()
        self.model = model
        self.image_path = image_path
        self.image_tensor = self.image_to_tensor()
        self.result = -1 # class 1 = 0, class 2 = 1, no result = -1 
    
    def __str__(self):
        return f"{self.timestamp} - ({self.result})"

    def image_to_tensor(self):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        
        image = read_image(self.image_path)
        image = transform(image)
        image = image.unsqueeze(0)

        return image
    
    def forward(self):
        device = next(self.model.parameters()).device
        self.image_tensor = self.image_tensor.to(device)

        with torch.no_grad():
            prediction = torch.sigmoid(self.model(self.image_tensor))

        predicted_class = -1

        if prediction.item() < 0.5:
            predicted_class = 0 #class1
        else:
            predicted_class = 1 #class2

        self.result = predicted_class

        return predicted_class
