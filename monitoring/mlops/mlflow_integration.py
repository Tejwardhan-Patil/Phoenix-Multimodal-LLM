import mlflow
import mlflow.pyfunc
from mlflow.models.signature import infer_signature
import os
import logging
import time
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MLflow_Integration")

# Constants
EXPERIMENT_NAME = "Phoenix_Multimodal_Model_Experiment"
MODEL_NAME = "Phoenix_Multimodal_Model"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MAX_EPOCHS = 50

# Initialize MLflow experiment
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# Define a multimodal model class
class MultimodalModel(mlflow.pyfunc.PythonModel):

    def __init__(self, model_params):
        self.model_params = model_params
        self.weights = None

    def train(self, train_data, epochs=MAX_EPOCHS):
        logger.info("Starting model training...")
        self.weights = np.random.rand(10) 
        for epoch in range(epochs):
            loss = np.random.random()
            accuracy = np.random.random()
            f1 = np.random.random()
            mlflow.log_metric("loss", loss, step=epoch)
            mlflow.log_metric("accuracy", accuracy, step=epoch)
            mlflow.log_metric("f1_score", f1, step=epoch)
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
            time.sleep(0.1)  # Training time

        logger.info("Model training completed.")

    def predict(self, context, model_input):
        logger.info("Running inference...")
        result = np.dot(model_input, self.weights)
        return result

# Training and validation data
def load_training_data():
    logger.info("Loading training data...")
    train_data = np.random.rand(100, 10)  # 100 samples, 10 features
    train_labels = np.random.randint(0, 2, 100)
    return train_data, train_labels

def load_validation_data():
    logger.info("Loading validation data...")
    val_data = np.random.rand(20, 10)  # 20 samples, 10 features
    val_labels = np.random.randint(0, 2, 20)
    return val_data, val_labels

# Function to log model parameters and metadata
def log_model_params(params):
    logger.info(f"Logging Parameters: {params}")
    mlflow.log_params(params)

# Function to log model signature (input-output structure)
def log_model_signature(train_data, train_labels):
    signature = infer_signature(train_data, train_labels)
    mlflow.log_text(str(signature), "model_signature.txt")
    logger.info(f"Model signature logged: {signature}")
    return signature

# Function to log and track training time
def log_training_time(start_time):
    elapsed_time = time.time() - start_time
    mlflow.log_metric("training_duration", elapsed_time)
    logger.info(f"Training completed in {elapsed_time:.2f} seconds.")

# Function to evaluate model
def evaluate_model(model, val_data, val_labels):
    logger.info("Evaluating model on validation set...")
    predictions = model.predict(None, val_data)
    accuracy = accuracy_score(val_labels, predictions.round())
    f1 = f1_score(val_labels, predictions.round(), average='macro')
    logger.info(f"Validation Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
    mlflow.log_metric("val_accuracy", accuracy)
    mlflow.log_metric("val_f1_score", f1)

# Register the model for future deployment
def register_model(run_id, model_name):
    logger.info(f"Registering model {model_name}...")
    mlflow.register_model(
        model_uri=f"runs:/{run_id}/multimodal_model",
        name=model_name
    )
    logger.info(f"Model {model_name} registered successfully.")

# Main function to execute the training pipeline
def run_pipeline():
    try:
        logger.info("Starting MLflow run...")
        with mlflow.start_run(run_name="Phoenix_Model_Run") as run:
            start_time = time.time()

            # Load training and validation data
            train_data, train_labels = load_training_data()
            val_data, val_labels = load_validation_data()

            # Define and log model parameters
            model_params = {"learning_rate": 0.001, "batch_size": 32, "modality": "multimodal"}
            log_model_params(model_params)

            # Initialize and train the model
            multimodal_model = MultimodalModel(model_params)
            multimodal_model.train(train_data)

            # Log model signature
            model_signature = log_model_signature(train_data, train_labels)

            # Evaluate the model
            evaluate_model(multimodal_model, val_data, val_labels)

            # Log training duration
            log_training_time(start_time)

            # Save and register the model
            mlflow.pyfunc.log_model(
                artifact_path="multimodal_model",
                python_model=multimodal_model,
                signature=model_signature
            )
            register_model(run.info.run_id, MODEL_NAME)

    except Exception as e:
        logger.error(f"Error during MLflow pipeline execution: {str(e)}")
        mlflow.log_artifact("error_log.txt")

# Set up tracking URI and environment
def setup_environment():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"Tracking URI set to {tracking_uri}")
    else:
        logger.info(f"Using default tracking URI: {MLFLOW_TRACKING_URI}")

# Load a registered model from the MLflow model registry
def load_registered_model(model_name, version=None):
    try:
        logger.info(f"Loading model {model_name}, version {version}")
        model_uri = f"models:/{model_name}/{version}" if version else f"models:/{model_name}/latest"
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"Model {model_name} loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {str(e)}")
        return None

# Inference on new data
def inference_on_new_data(model, input_data):
    if model:
        logger.info("Running inference on new data...")
        predictions = model.predict(None, input_data)
        logger.info(f"Inference results: {predictions}")
        return predictions
    else:
        logger.error("No model available for inference.")
        return None

# Entry point for the script
if __name__ == "__main__":
    setup_environment()

    # Run the pipeline (training, logging, evaluation, and registration)
    run_pipeline()

    # Load and run inference with the registered model
    registered_model = load_registered_model(MODEL_NAME, version=1)

    # Data for inference
    new_data = np.random.rand(10, 10)
    inference_results = inference_on_new_data(registered_model, new_data)