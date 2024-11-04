import os
import logging
from azureml.core import Workspace, Model, Environment, ComputeTarget
from azureml.core.webservice import AksWebservice, Webservice
from azureml.core.model import InferenceConfig
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.compute import AksCompute
from azureml.exceptions import ComputeTargetException

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_workspace():
    """Retrieve the Azure ML workspace."""
    try:
        subscription_id = os.getenv('AZURE_SUBSCRIPTION_ID')
        resource_group = os.getenv('AZURE_RESOURCE_GROUP')
        workspace_name = os.getenv('AZURE_WORKSPACE_NAME')

        workspace = Workspace(subscription_id=subscription_id,
                              resource_group=resource_group,
                              workspace_name=workspace_name)
        logger.info("Workspace loaded successfully.")
        return workspace
    except Exception as e:
        logger.error(f"Failed to load workspace: {str(e)}")
        raise

def get_model(workspace, model_name):
    """Retrieve the registered model."""
    try:
        model = Model(workspace, model_name)
        logger.info(f"Model '{model_name}' loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

def setup_environment():
    """Set up the environment for the model."""
    try:
        env = Environment(name="multimodal-env")
        conda_dep = CondaDependencies()

        # Adding required packages
        conda_dep.add_pip_package("azureml-core")
        conda_dep.add_pip_package("torch")
        conda_dep.add_pip_package("transformers")
        conda_dep.add_pip_package("fastapi")
        conda_dep.add_pip_package("uvicorn")

        env.python.conda_dependencies = conda_dep
        logger.info("Environment setup with required dependencies.")
        return env
    except Exception as e:
        logger.error(f"Failed to set up environment: {str(e)}")
        raise

def create_inference_config(env):
    """Set up the inference configuration."""
    try:
        inference_config = InferenceConfig(entry_script="inference.py", environment=env)
        logger.info("Inference configuration set up successfully.")
        return inference_config
    except Exception as e:
        logger.error(f"Failed to set up inference configuration: {str(e)}")
        raise

def get_aks_target(workspace, aks_name):
    """Retrieve or create an AKS compute target."""
    try:
        aks_target = ComputeTarget(workspace=workspace, name=aks_name)
        logger.info(f"Using existing AKS compute target '{aks_name}'.")
    except ComputeTargetException:
        logger.info(f"Creating new AKS compute target '{aks_name}'.")
        try:
            aks_config = AksCompute.provisioning_configuration(agent_count=3, vm_size='Standard_D3_v2')
            aks_target = ComputeTarget.create(workspace, aks_name, aks_config)
            aks_target.wait_for_completion(show_output=True)
            logger.info(f"AKS compute target '{aks_name}' created successfully.")
        except Exception as e:
            logger.error(f"Failed to create AKS compute target: {str(e)}")
            raise
    return aks_target

def deploy_model(workspace, model, inference_config, aks_target, deployment_config):
    """Deploy the model to the specified target."""
    try:
        service_name = f"{model.name}-service"
        logger.info(f"Deploying model '{model.name}' to AKS target.")

        service = Model.deploy(workspace=workspace,
                               name=service_name,
                               models=[model],
                               inference_config=inference_config,
                               deployment_config=deployment_config,
                               deployment_target=aks_target)

        service.wait_for_deployment(show_output=True)
        logger.info(f"Model deployed successfully as service '{service_name}'.")
        return service
    except Exception as e:
        logger.error(f"Model deployment failed: {str(e)}")
        raise

def configure_deployment(cpu_cores=2, memory_gb=8):
    """Configure the deployment settings."""
    try:
        deployment_config = AksWebservice.deploy_configuration(cpu_cores=cpu_cores,
                                                               memory_gb=memory_gb,
                                                               enable_app_insights=True)
        logger.info("Deployment configuration created successfully.")
        return deployment_config
    except Exception as e:
        logger.error(f"Failed to configure deployment: {str(e)}")
        raise

def output_service_details(service):
    """Output the deployed service details."""
    try:
        logger.info(f"Service state: {service.state}")
        logger.info(f"Scoring URI: {service.scoring_uri}")
        logger.info(f"Swagger URI: {service.swagger_uri}")
    except Exception as e:
        logger.error(f"Failed to retrieve service details: {str(e)}")
        raise

def main():
    """Main function to orchestrate model deployment."""
    try:
        workspace = get_workspace()

        model_name = os.getenv('MODEL_NAME')
        aks_name = os.getenv('AKS_CLUSTER_NAME')

        model = get_model(workspace, model_name)
        environment = setup_environment()
        inference_config = create_inference_config(environment)

        aks_target = get_aks_target(workspace, aks_name)
        deployment_config = configure_deployment(cpu_cores=2, memory_gb=8)

        service = deploy_model(workspace, model, inference_config, aks_target, deployment_config)
        output_service_details(service)

    except Exception as e:
        logger.error(f"Deployment process failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()