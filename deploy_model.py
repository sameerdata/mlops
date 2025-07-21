from azureml.core import Workspace, Model, Environment
from azureml.core.webservice import AciWebservice, Webservice
from azureml.core.model import InferenceConfig
from azureml.exceptions import WebserviceException

# Load workspace
ws = Workspace.from_config()

# Delete existing endpoint if exists
service_name = "churn-endpoint"
try:
    old_service = Webservice(ws, name=service_name)
    old_service.delete()
    print("🧹 Old endpoint deleted.")
except WebserviceException:
    print("ℹ️ No existing endpoint found.")

# Define environment and ensure azureml-defaults is included
env = Environment.from_pip_requirements(name='ml-env', file_path='src/requirements.txt')
env.python.conda_dependencies.add_pip_package("azureml-defaults")

# Define inference configuration
inference_config = InferenceConfig(entry_script="src/score.py", environment=env)

# Define deployment configuration
deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

# Deploy model
model = Model(ws, name="churn_model")

service = Model.deploy(
    workspace=ws,
    name=service_name,
    models=[model],
    inference_config=inference_config,
    deployment_config=deployment_config,
    overwrite=True
)

# Wait and show logs if failed
service.wait_for_deployment(show_output=True)

if service.state != "Healthy":
    print("⚠️ Deployment failed. Fetching logs:")
    print(service.get_logs())

print(f"✅ Deployed at: {service.scoring_uri}")
