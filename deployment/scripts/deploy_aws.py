import boto3
import os
import time
import json
import paramiko
import logging
from botocore.exceptions import NoCredentialsError, ClientError

# Logger setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# AWS Configuration
AWS_REGION = "us-west-2"
EC2_INSTANCE_TYPE = "ml.g4dn.xlarge"
S3_BUCKET_NAME = "multimodal-model-bucket"
MODEL_S3_PATH = "s3://multimodal-model-bucket/models/latest_model.tar.gz"
IAM_ROLE = "arn:aws:iam::123456789012:role/EC2DeployRole"
KEY_PAIR_NAME = "aws-key-pair"
LOCAL_KEY_PATH = "/aws-key-pair.pem"
SECURITY_GROUP_ID = "sg-0123456789abcdef0"
IMAGE_ID = "ami-0abcdef1234567890" 
INSTANCE_TAG = {"Key": "Project", "Value": "PhoenixMultimodal"}

# SSH Configuration
SSH_PORT = 22
SSH_USERNAME = "ec2-user"

# EC2 client
session = boto3.Session(region_name=AWS_REGION)
ec2_client = session.client('ec2')
s3_client = session.client('s3')
ssm_client = session.client('ssm')

# Functions for model deployment
def upload_model_to_s3(local_model_path):
    """Uploads the model to the specified S3 bucket."""
    try:
        logger.info("Starting upload of model to S3...")
        s3_client.upload_file(local_model_path, S3_BUCKET_NAME, "models/latest_model.tar.gz")
        logger.info(f"Model uploaded successfully to {MODEL_S3_PATH}")
    except FileNotFoundError:
        logger.error("Model file not found. Please verify the path.")
    except NoCredentialsError:
        logger.error("AWS credentials not available.")
    except Exception as e:
        logger.error(f"Error uploading model: {e}")

def create_ec2_instance():
    """Creates an EC2 instance for model deployment."""
    try:
        logger.info("Launching EC2 instance...")
        response = ec2_client.run_instances(
            ImageId=IMAGE_ID,
            InstanceType=EC2_INSTANCE_TYPE,
            KeyName=KEY_PAIR_NAME,
            MinCount=1,
            MaxCount=1,
            IamInstanceProfile={'Arn': IAM_ROLE},
            SecurityGroupIds=[SECURITY_GROUP_ID],
            TagSpecifications=[{
                'ResourceType': 'instance',
                'Tags': [INSTANCE_TAG]
            }],
            UserData=USER_DATA_SCRIPT
        )
        instance_id = response['Instances'][0]['InstanceId']
        logger.info(f"EC2 instance {instance_id} created successfully.")
        return instance_id
    except ClientError as e:
        logger.error(f"Failed to create EC2 instance: {e}")
        return None

def wait_for_instance(instance_id):
    """Waits for the EC2 instance to reach the running state."""
    logger.info(f"Waiting for instance {instance_id} to be running...")
    try:
        ec2_client.get_waiter('instance_running').wait(InstanceIds=[instance_id])
        logger.info(f"Instance {instance_id} is now running.")
    except ClientError as e:
        logger.error(f"Error while waiting for instance {instance_id} to start: {e}")

def get_instance_public_ip(instance_id):
    """Retrieves the public IP of the EC2 instance."""
    try:
        response = ec2_client.describe_instances(InstanceIds=[instance_id])
        ip_address = response['Reservations'][0]['Instances'][0]['PublicIpAddress']
        logger.info(f"Instance {instance_id} has public IP {ip_address}")
        return ip_address
    except ClientError as e:
        logger.error(f"Error retrieving public IP for instance {instance_id}: {e}")
        return None

def ssh_connect(instance_ip):
    """Establishes SSH connection to the EC2 instance."""
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        logger.info(f"Attempting to connect to instance at {instance_ip} via SSH...")
        ssh.connect(instance_ip, port=SSH_PORT, username=SSH_USERNAME, key_filename=LOCAL_KEY_PATH)
        logger.info(f"SSH connection established with {instance_ip}.")
        return ssh
    except Exception as e:
        logger.error(f"SSH connection failed: {e}")
        return None

def execute_remote_command(ssh_client, command):
    """Executes a command on the remote EC2 instance via SSH."""
    try:
        logger.info(f"Executing remote command: {command}")
        stdin, stdout, stderr = ssh_client.exec_command(command)
        output = stdout.read().decode('utf-8')
        error = stderr.read().decode('utf-8')
        if output:
            logger.info(f"Command output: {output}")
        if error:
            logger.error(f"Command error: {error}")
    except Exception as e:
        logger.error(f"Failed to execute command {command}: {e}")

def deploy_model_to_ec2(instance_ip):
    """Deploys the multimodal model to the EC2 instance."""
    logger.info(f"Starting model deployment on EC2 instance with IP {instance_ip}...")
    ssh_client = ssh_connect(instance_ip)
    if ssh_client:
        execute_remote_command(ssh_client, "mkdir -p /opt/multimodal_model")
        execute_remote_command(ssh_client, f"aws s3 cp {MODEL_S3_PATH} /opt/multimodal_model/")
        execute_remote_command(ssh_client, "cd /opt/multimodal_model && tar -xzvf latest_model.tar.gz")
        execute_remote_command(ssh_client, "docker-compose up -d")
        ssh_client.close()

def terminate_instance(instance_id):
    """Terminates the EC2 instance after deployment."""
    try:
        logger.info(f"Terminating instance {instance_id}...")
        ec2_client.terminate_instances(InstanceIds=[instance_id])
        ec2_client.get_waiter('instance_terminated').wait(InstanceIds=[instance_id])
        logger.info(f"Instance {instance_id} terminated.")
    except ClientError as e:
        logger.error(f"Error terminating instance {instance_id}: {e}")

USER_DATA_SCRIPT = """#!/bin/bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo service docker start
docker pull mymultimodal/model:latest
docker run -d -p 8080:8080 mymultimodal/model:latest
"""

def main():
    """Main function for deploying model on AWS."""
    # Step 1: Upload model to S3
    upload_model_to_s3('/local/model.tar.gz')

    # Step 2: Create EC2 instance
    instance_id = create_ec2_instance()
    if not instance_id:
        logger.error("Instance creation failed. Exiting...")
        return

    # Step 3: Wait for instance to start
    wait_for_instance(instance_id)

    # Step 4: Get instance public IP
    instance_ip = get_instance_public_ip(instance_id)
    if not instance_ip:
        logger.error("Failed to retrieve instance public IP. Exiting...")
        return

    # Step 5: Deploy model on EC2
    deploy_model_to_ec2(instance_ip)

    # Step 6: Terminate instance
    terminate_instance(instance_id)

if __name__ == "__main__":
    main()