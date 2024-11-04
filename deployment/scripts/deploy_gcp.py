import os
import google.auth
from google.cloud import storage
from googleapiclient.discovery import build
from google.oauth2 import service_account
from google.cloud import compute_v1
import time
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load Google Cloud credentials and define constants
credentials, project = google.auth.default()
ZONE = "us-central1-a"
INSTANCE_NAME = "multimodal-model-instance"
MACHINE_TYPE = "n1-standard-4"
IMAGE_PROJECT = "debian-cloud"
IMAGE_FAMILY = "debian-10"
BUCKET_NAME = "multimodal-model-bucket"
MODEL_NAME = "multimodal_model.tar.gz"
STARTUP_SCRIPT = f"""#!/bin/bash
sudo apt-get update
sudo apt-get install -y python3-pip
gsutil cp gs://{BUCKET_NAME}/{MODEL_NAME} /home/{MODEL_NAME}
tar -xvzf /home/{MODEL_NAME} -C /home/
cd /home/multimodal_model
pip3 install -r requirements.txt
python3 app.py
"""

# Function to create a storage bucket, with error handling
def create_bucket(bucket_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    try:
        if not bucket.exists():
            bucket = storage_client.create_bucket(bucket_name)
            logging.info(f"Bucket {bucket_name} created successfully.")
        else:
            logging.info(f"Bucket {bucket_name} already exists.")
    except Exception as e:
        logging.error(f"Error creating bucket {bucket_name}: {str(e)}")
    return bucket

# Function to upload model file to GCP bucket with retries
def upload_model(bucket_name, model_path, retries=3):
    bucket = create_bucket(bucket_name)
    blob = bucket.blob(os.path.basename(model_path))
    attempt = 0
    while attempt < retries:
        try:
            blob.upload_from_filename(model_path)
            logging.info(f"Model {model_path} uploaded to bucket {bucket_name}.")
            break
        except Exception as e:
            attempt += 1
            logging.error(f"Attempt {attempt}: Error uploading model: {str(e)}")
            time.sleep(2 ** attempt)  # Exponential backoff
    if attempt == retries:
        logging.error(f"Failed to upload model {model_path} after {retries} attempts.")

# Function to create GCP compute instance with error handling and retries
def create_instance(compute, project, zone, instance_name, bucket_name):
    try:
        image_response = compute.images().getFromFamily(
            project=IMAGE_PROJECT, family=IMAGE_FAMILY).execute()
        source_disk_image = image_response['selfLink']
        
        machine_type = f"zones/{zone}/machineTypes/{MACHINE_TYPE}"
        config = {
            "name": instance_name,
            "machineType": machine_type,
            "disks": [{
                "boot": True,
                "autoDelete": True,
                "initializeParams": {
                    "sourceImage": source_disk_image,
                }
            }],
            "networkInterfaces": [{
                "network": "global/networks/default",
                "accessConfigs": [{"type": "ONE_TO_ONE_NAT", "name": "External NAT"}]
            }],
            "metadata": {
                "items": [{
                    "key": "startup-script",
                    "value": STARTUP_SCRIPT
                }]
            }
        }

        operation = compute.instances().insert(
            project=project,
            zone=zone,
            body=config).execute()

        logging.info(f"Instance {instance_name} creation initiated in zone {zone}.")
        return operation

    except Exception as e:
        logging.error(f"Error creating instance {instance_name}: {str(e)}")
        raise

# Function to monitor the progress of an operation
def wait_for_operation(compute, project, zone, operation):
    logging.info(f"Waiting for operation {operation['name']} to finish...")
    while True:
        result = compute.zoneOperations().get(
            project=project,
            zone=zone,
            operation=operation['name']).execute()

        if result['status'] == 'DONE':
            if 'error' in result:
                logging.error(f"Operation {operation['name']} failed with error: {result['error']}")
                raise Exception(result['error'])
            logging.info(f"Operation {operation['name']} completed successfully.")
            return result

        time.sleep(5)

# Function to deploy model on GCP by creating a bucket, uploading model, and creating instance
def deploy_model_gcp(model_path):
    # Upload model to GCP bucket
    logging.info(f"Starting the deployment process for {MODEL_NAME}.")
    upload_model(BUCKET_NAME, model_path)

    # Initialize compute engine client
    compute = build('compute', 'v1', credentials=credentials)

    # Create a new instance for the model
    logging.info(f"Creating instance {INSTANCE_NAME} in zone {ZONE}.")
    operation = create_instance(compute, project, ZONE, INSTANCE_NAME, BUCKET_NAME)

    # Wait for the instance to be created
    wait_for_operation(compute, project, ZONE, operation)

# Additional helper function to delete an instance if needed
def delete_instance(compute, project, zone, instance_name):
    try:
        logging.info(f"Deleting instance {instance_name}.")
        operation = compute.instances().delete(
            project=project,
            zone=zone,
            instance=instance_name).execute()
        wait_for_operation(compute, project, zone, operation)
        logging.info(f"Instance {instance_name} deleted successfully.")
    except Exception as e:
        logging.error(f"Error deleting instance {instance_name}: {str(e)}")

# Function to delete the storage bucket
def delete_bucket(bucket_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    try:
        bucket.delete(force=True)
        logging.info(f"Bucket {bucket_name} deleted successfully.")
    except Exception as e:
        logging.error(f"Error deleting bucket {bucket_name}: {str(e)}")

# Function to clean up resources (instance and bucket) after deployment
def clean_up_resources():
    compute = build('compute', 'v1', credentials=credentials)
    delete_instance(compute, project, ZONE, INSTANCE_NAME)
    delete_bucket(BUCKET_NAME)

# Enhanced logging for deployment status tracking
def log_deployment_status():
    logging.info(f"Deployment of {MODEL_NAME} to GCP started.")
    try:
        deploy_model_gcp("/multimodal_model.tar.gz")
        logging.info(f"Deployment of {MODEL_NAME} completed successfully.")
    except Exception as e:
        logging.error(f"Deployment failed: {str(e)}")
        clean_up_resources()

# Usage
if __name__ == "__main__":
    try:
        log_deployment_status()
    except Exception as e:
        logging.critical(f"Unexpected error during deployment: {str(e)}")
        clean_up_resources()