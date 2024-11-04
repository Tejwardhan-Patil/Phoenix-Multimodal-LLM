import logging
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Optional
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.text.gpt_model import GPTModel
from models.image.cnn_model import CNNModel
from models.audio.wavenet_model import WaveNetModel
from models.multimodal.fusion_model import FusionModel

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

# Load models
gpt_model = GPTModel()
cnn_model = CNNModel()
wavenet_model = WaveNetModel()
fusion_model = FusionModel()

# Input Models
class TextInput(BaseModel):
    text: str = Field(..., title="Text", description="Input text for prediction", min_length=1, max_length=1000)

class ImageInput(BaseModel):
    image_path: str = Field(..., title="Image Path", description="Path to the image file for prediction")

class AudioInput(BaseModel):
    audio_path: str = Field(..., title="Audio Path", description="Path to the audio file for prediction")

class MultimodalInput(BaseModel):
    text: str = Field(..., title="Text", description="Text input for multimodal prediction", min_length=1, max_length=1000)
    image_path: str = Field(..., title="Image Path", description="Path to the image file for multimodal prediction")
    audio_path: str = Field(..., title="Audio Path", description="Path to the audio file for multimodal prediction")

class BatchTextInput(BaseModel):
    texts: List[str] = Field(..., title="Texts", description="Batch of text inputs for prediction", min_items=1, max_items=100)

# Utility function for logging request and response details
def log_request(request: Request, input_data: dict):
    logger.info(f"Request: {request.method} {request.url}")
    logger.info(f"Input data: {input_data}")

def log_response(response: dict):
    logger.info(f"Response: {response}")

# Health Check Endpoint
@app.get("/health")
async def health_check():
    logger.info("Health check endpoint accessed.")
    return {"status": "healthy"}

# Text Prediction Endpoint
@app.post("/predict/text")
async def predict_text(input: TextInput, request: Request):
    log_request(request, input.dict())
    try:
        prediction = gpt_model.predict(input.text)
        response = {"prediction": prediction, "model": "GPT"}
        log_response(response)
        return response
    except Exception as e:
        logger.error(f"Error during text prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text prediction failed: {str(e)}")

# Batch Text Prediction Endpoint
@app.post("/predict/text/batch")
async def predict_batch_text(input: BatchTextInput, request: Request):
    log_request(request, input.dict())
    try:
        predictions = [gpt_model.predict(text) for text in input.texts]
        response = {"predictions": predictions, "model": "GPT"}
        log_response(response)
        return response
    except Exception as e:
        logger.error(f"Error during batch text prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch text prediction failed: {str(e)}")

# Image Prediction Endpoint
@app.post("/predict/image")
async def predict_image(input: ImageInput, request: Request):
    log_request(request, input.dict())
    if not os.path.exists(input.image_path):
        logger.error(f"Image file not found: {input.image_path}")
        raise HTTPException(status_code=404, detail="Image file not found.")
    
    try:
        prediction = cnn_model.predict(input.image_path)
        response = {"prediction": prediction, "model": "CNN"}
        log_response(response)
        return response
    except Exception as e:
        logger.error(f"Error during image prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image prediction failed: {str(e)}")

# Audio Prediction Endpoint
@app.post("/predict/audio")
async def predict_audio(input: AudioInput, request: Request):
    log_request(request, input.dict())
    if not os.path.exists(input.audio_path):
        logger.error(f"Audio file not found: {input.audio_path}")
        raise HTTPException(status_code=404, detail="Audio file not found.")
    
    try:
        prediction = wavenet_model.predict(input.audio_path)
        response = {"prediction": prediction, "model": "WaveNet"}
        log_response(response)
        return response
    except Exception as e:
        logger.error(f"Error during audio prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Audio prediction failed: {str(e)}")

# Multimodal Prediction Endpoint
@app.post("/predict/multimodal")
async def predict_multimodal(input: MultimodalInput, request: Request):
    log_request(request, input.dict())
    
    if not os.path.exists(input.image_path):
        logger.error(f"Image file not found: {input.image_path}")
        raise HTTPException(status_code=404, detail="Image file not found.")
    
    if not os.path.exists(input.audio_path):
        logger.error(f"Audio file not found: {input.audio_path}")
        raise HTTPException(status_code=404, detail="Audio file not found.")
    
    try:
        prediction = fusion_model.predict(input.text, input.image_path, input.audio_path)
        response = {"prediction": prediction, "model": "Multimodal Fusion"}
        log_response(response)
        return response
    except Exception as e:
        logger.error(f"Error during multimodal prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Multimodal prediction failed: {str(e)}")

# Get available models
@app.get("/models")
async def get_models():
    models = {
        "text_model": "GPT",
        "image_model": "CNN",
        "audio_model": "WaveNet",
        "multimodal_model": "Fusion Model"
    }
    logger.info(f"Models available: {models}")
    return models

# Additional Logging for requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Completed request: {request.method} {request.url} - Status: {response.status_code}")
    return response

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)