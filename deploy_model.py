from azureml.core import Workspace, Model, Environment
from azureml.core.webservice import AciWebservice, Webservice
from azureml.core.model import InferenceConfig

# Load workspace
ws = Workspace.from_config()

# Delete existing endpoint if exists
service_name = "churn-endpoint"
try:
    old_service = Webservice(ws, name=service_name)
    old_service.delete()
    print("🧹 Old endpoint deleted.")
except Exception:
    print("ℹ️ No existing endpoint found.")

# Define environment
env = Environment.from_pip_requirements(name='ml-env', file_path='src/requirements.txt')

# Inference config
inference_config = InferenceConfig(entry_script="src/score.py", environment=env)

# Deployment config
deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

# Deploy model
service = Model.deploy(
    workspace=ws,
    name=service_name,
    models=[Model(ws, "churn_model")],
    inference_config=inference_config,
    deployment_config=deployment_config,
    overwrite=True
)
service.wait_for_deployment(show_output=True)

print(f"✅ Deployed at: {service.scoring_uri}")
