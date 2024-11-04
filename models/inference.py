import torch
import logging
from text.gpt_model import GPTModel
from image.cnn_model import CNNModel
from audio.rnn_audio_model import RNNAudioModel
from multimodal.fusion_model import FusionModel
from utils.data_loader import load_text_data, load_image_data, load_audio_data
from utils.metrics import calculate_metrics
from utils.visualization import visualize_predictions

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load models for each modality
text_model = GPTModel()
image_model = CNNModel()
audio_model = RNNAudioModel()
fusion_model = FusionModel()

# Load pre-trained model weights
def load_model_weights():
    try:
        text_model.load_state_dict(torch.load('pretrained/text_model.pth'))
        image_model.load_state_dict(torch.load('pretrained/image_model.pth'))
        audio_model.load_state_dict(torch.load('pretrained/audio_model.pth'))
        fusion_model.load_state_dict(torch.load('pretrained/fusion_model.pth'))
        logger.info("Model weights loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model weights: {e}")
        raise

# Set models to evaluation mode
def set_eval_mode():
    text_model.eval()
    image_model.eval()
    audio_model.eval()
    fusion_model.eval()
    logger.info("Models set to evaluation mode.")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
text_model.to(device)
image_model.to(device)
audio_model.to(device)
fusion_model.to(device)
logger.info(f"Running on device: {device}")

# Validate input formats
def validate_input(text_input, image_input, audio_input):
    if not isinstance(text_input, str):
        raise ValueError("Text input must be a string.")
    if not isinstance(image_input, str):
        raise ValueError("Image input must be a file path as a string.")
    if not isinstance(audio_input, str):
        raise ValueError("Audio input must be a file path as a string.")

# Run inference on multimodal input
def run_inference(text_input, image_input, audio_input):
    validate_input(text_input, image_input, audio_input)
    
    # Load and preprocess inputs
    try:
        text_data = load_text_data(text_input)
        image_data = load_image_data(image_input)
        audio_data = load_audio_data(audio_input)
        logger.info("Inputs loaded and preprocessed successfully.")
    except Exception as e:
        logger.error(f"Error in data loading/preprocessing: {e}")
        raise

    # Move data to the device
    text_data = text_data.to(device)
    image_data = image_data.to(device)
    audio_data = audio_data.to(device)

    # Run individual models and handle potential errors
    try:
        text_output = text_model(text_data)
        image_output = image_model(image_data)
        audio_output = audio_model(audio_data)
        logger.info("Model outputs generated successfully.")
    except Exception as e:
        logger.error(f"Error during model inference: {e}")
        raise

    # Fuse outputs
    try:
        fused_output = fusion_model(text_output, image_output, audio_output)
        logger.info("Fusion of model outputs completed successfully.")
    except Exception as e:
        logger.error(f"Error during fusion: {e}")
        raise

    # Postprocess predictions
    predictions = torch.argmax(fused_output, dim=1)
    logger.info(f"Predictions generated: {predictions}")

    return predictions

# Calculate metrics
def evaluate_model(predictions, true_labels):
    try:
        metrics = calculate_metrics(predictions, true_labels)
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        raise

# Visualize predictions
def visualize_output(predictions, inputs):
    try:
        visualize_predictions(predictions, inputs)
        logger.info("Visualization generated successfully.")
    except Exception as e:
        logger.error(f"Error during visualization: {e}")
        raise

# Inference function with detailed logging and metrics tracking
def multimodal_inference_pipeline(text_input, image_input, audio_input, true_labels=None):
    try:
        logger.info("Starting multimodal inference pipeline.")
        
        # Load model weights
        load_model_weights()
        
        # Set models to evaluation mode
        set_eval_mode()
        
        # Run inference
        predictions = run_inference(text_input, image_input, audio_input)
        
        # If true labels are provided, calculate metrics
        if true_labels is not None:
            metrics = evaluate_model(predictions, true_labels)
        else:
            logger.info("True labels not provided, skipping metrics calculation.")
        
        # Visualize the predictions
        visualize_output(predictions, [text_input, image_input, audio_input])
        
        logger.info("Multimodal inference pipeline completed successfully.")
        return predictions
    except Exception as e:
        logger.error(f"Error in inference pipeline: {e}")
        raise

# Inference with logging and error handling
if __name__ == "__main__":
    # Sample inputs
    text_input = "Sample text for inference"
    image_input = "/sample_image.jpg"
    audio_input = "/sample_audio.wav"

    # True labels for evaluation
    true_labels = torch.tensor([0]) 

    # Run the inference pipeline
    try:
        predictions = multimodal_inference_pipeline(text_input, image_input, audio_input, true_labels)
        logger.info(f"Final Predictions: {predictions}")
    except Exception as e:
        logger.error(f"Failed to complete inference: {e}")