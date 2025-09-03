from fastapi import FastAPI, File
from PIL import Image
from fastapi.testclient import TestClient
from main import app
import io 

client = TestClient(app)

def make_test_image():
    '''generates test input image. not for testing correct predcition from model. 
    Only suffices to test functionality of API.
    avoids I/O by not creating file of the image by just loading it into a memory buffer'''
    img = Image.new("RGB", (64, 64), color=(161, 161, 161))
    buffer = io.BytesIO() # create buffer in memory where the image will live
    img.save(buffer, format="PNG") # save image to the buffer
    buffer.seek(0) # rewind buffer pointer to beginning of buffer before returning

    return buffer

def test_responselist():
    ''' test that the list of responses is shown'''
    responselist = client.get("/predict/")
    assert responselist.status_code == 200 # list of responses requested successfully

def test_predict():
    ''' test that prediction forward pass is working and response has the correct format'''
    buffer = make_test_image() #get buffer filled with test image
    data = {"name": "test"}
    files = {"img": ("test.png", buffer, "image/png") }
    response = client.post("/predict", data=data, files=files)
    response_json = response.json()
    assert response.status_code == 200
    assert response_json["name"] == "test"
    assert response_json["filename"] == "test.png"
    assert response_json["result"] in [0,1]



