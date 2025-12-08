from fastapi import FastAPI
from fastapi import FastAPI, File, UploadFile
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

MODEL= tf.keras.models.load_model("app/model.keras")
class_names=['Potato_Early_blight', 'Potato_Late_blight', 'Potato_healthy']

app =FastAPI()
@app.get("/ping")
async def ping():
    return "hello world"


def read_file_as_image(data) -> np.ndarray:
    image=Image.open(BytesIO(data)).resize((256,256))
    return np.array(image)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    
    image= read_file_as_image( await file.read())
    batch_image=np.expand_dims(image,0)
    prediction=MODEL.predict(batch_image)
    predicted_class=class_names[np.argmax(prediction[0])]
    confidence=float(np.max(prediction[0]))
    return {
        'class':predicted_class,
        'confidence':confidence
    }
