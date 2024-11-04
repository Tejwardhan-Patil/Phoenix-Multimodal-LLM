from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
from deployment.api.app import multimodal_model_inference
import logging

router = APIRouter()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Exception classes
class ModelInferenceError(Exception):
    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(self.detail)

class InvalidInputError(Exception):
    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(self.detail)

# Input Schemas
class TextInput(BaseModel):
    text: str = Field(..., min_length=1, description="Text input for the model")

class AudioInput(BaseModel):
    audio_data: bytes

class ImageInput(BaseModel):
    image_data: bytes

# Output Schemas
class InferenceOutput(BaseModel):
    result: dict
    status: str

# Utility Functions
def log_inference(text: Optional[str], image: Optional[bytes], audio: Optional[bytes]):
    logger.info("Inference requested")
    if text:
        logger.info(f"Text input: {text}")
    if image:
        logger.info(f"Image input: {len(image)} bytes")
    if audio:
        logger.info(f"Audio input: {len(audio)} bytes")

def validate_inference_output(output: dict):
    if not isinstance(output, dict) or 'result' not in output:
        raise ModelInferenceError(detail="Invalid output format from model inference")

# API Routes
@router.post("/inference/text", response_model=InferenceOutput)
async def text_inference(input_data: TextInput):
    try:
        log_inference(text=input_data.text, image=None, audio=None)
        result = multimodal_model_inference(text=input_data.text, image=None, audio=None)
        validate_inference_output(result)
        return {"result": result, "status": "success"}
    except ModelInferenceError as e:
        logger.error(f"Model inference error: {e.detail}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=e.detail)
    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid input")

@router.post("/inference/audio", response_model=InferenceOutput)
async def audio_inference(audio: UploadFile = File(...)):
    try:
        audio_bytes = await audio.read()
        log_inference(text=None, image=None, audio=audio_bytes)
        if not audio_bytes:
            raise InvalidInputError(detail="Audio file is empty")
        result = multimodal_model_inference(text=None, image=None, audio=audio_bytes)
        validate_inference_output(result)
        return {"result": result, "status": "success"}
    except InvalidInputError as e:
        logger.error(f"Invalid input: {e.detail}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.detail)
    except ModelInferenceError as e:
        logger.error(f"Model inference error: {e.detail}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=e.detail)

@router.post("/inference/image", response_model=InferenceOutput)
async def image_inference(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        log_inference(text=None, image=image_bytes, audio=None)
        if not image_bytes:
            raise InvalidInputError(detail="Image file is empty")
        result = multimodal_model_inference(text=None, image=image_bytes, audio=None)
        validate_inference_output(result)
        return {"result": result, "status": "success"}
    except InvalidInputError as e:
        logger.error(f"Invalid input: {e.detail}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.detail)
    except ModelInferenceError as e:
        logger.error(f"Model inference error: {e.detail}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=e.detail)

@router.post("/inference/multimodal", response_model=InferenceOutput)
async def multimodal_inference(
    text: str = Form(...), 
    image: UploadFile = File(...), 
    audio: UploadFile = File(...)
):
    try:
        image_bytes = await image.read()
        audio_bytes = await audio.read()

        log_inference(text=text, image=image_bytes, audio=audio_bytes)
        if not text or not image_bytes or not audio_bytes:
            raise InvalidInputError(detail="Missing one or more required inputs")

        result = multimodal_model_inference(text=text, image=image_bytes, audio=audio_bytes)
        validate_inference_output(result)
        return {"result": result, "status": "success"}
    except InvalidInputError as e:
        logger.error(f"Invalid input: {e.detail}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.detail)
    except ModelInferenceError as e:
        logger.error(f"Model inference error: {e.detail}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=e.detail)

# Additional routes for health checks and metrics
@router.get("/health", status_code=200)
async def health_check():
    logger.info("Health check called")
    return {"status": "API is healthy"}

@router.get("/metrics", status_code=200)
async def api_metrics():
    logger.info("Metrics requested")
    metrics_data = {
        "requests": 10234,
        "average_latency": "120ms",
        "model_accuracy": "97.5%"
    }
    return metrics_data

# Error handlers
@router.exception_handler(ModelInferenceError)
async def model_inference_exception_handler(request, exc: ModelInferenceError):
    return JSONResponse(
        status_code=500,
        content={"message": exc.detail},
    )

@router.exception_handler(InvalidInputError)
async def invalid_input_exception_handler(request, exc: InvalidInputError):
    return JSONResponse(
        status_code=400,
        content={"message": exc.detail},
    )

# CORS and Middleware
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost",
    "https://website.com"
]

router.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Graceful shutdown and startup events
@router.on_event("startup")
async def startup_event():
    logger.info("API service starting up...")

@router.on_event("shutdown")
async def shutdown_event():
    logger.info("API service shutting down...")