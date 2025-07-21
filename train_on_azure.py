from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment, Dataset, Run
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.model import Model
import os

# Load workspace
ws = Workspace.from_config()

# Load dataset (TabularDataset)
dataset = Dataset.get_by_name(ws, 'customer_churn_dataset')

# Create or attach to compute cluster
cluster_name = "cpu-cluster"
if cluster_name in ws.compute_targets:
    compute_target = ws.compute_targets[cluster_name]
else:
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_DS11_V2', max_nodes=2)
    compute_target = ComputeTarget.create(ws, cluster_name, compute_config)
    compute_target.wait_for_completion(show_output=True)

# Define environment
env = Environment.from_pip_requirements(name='ml-env', file_path='src/requirements.txt')

# Convert dataset to CSV for use in training
print("📦 Converting Azure ML dataset to CSV file...")
df = dataset.to_pandas_dataframe()
csv_path = os.path.join('src', 'customer_churn_100.csv')
df.to_csv(csv_path, index=False)

# Configure training script
src = ScriptRunConfig(
    source_directory='src',
    script='train.py',
    arguments=['--data_path', 'customer_churn_100.csv'],
    compute_target=compute_target,
    environment=env
)

# Submit experiment
experiment = Experiment(workspace=ws, name='churn-training')
run = experiment.submit(src)
run.wait_for_completion(show_output=True)

# Download model from outputs
print("📥 Downloading model from AzureML run outputs...")
download_path = "src/sklearn_model.pkl"
try:
    run.download_file(name="outputs/sklearn_model.pkl", output_file_path=download_path)
    print(f"✅ Model downloaded to: {download_path}")
except Exception as e:
    print(f"❌ Failed to download model file: {e}")
    download_path = None

# Register model
if download_path and os.path.exists(download_path):
    print("📦 Registering trained model...")
    registered_model = Model.register(
        workspace=ws,
        model_path=download_path,
        model_name="churn_model"
    )
    print(f"✅ Model registered: {registered_model.name} (v{registered_model.version})")
else:
    print("❌ Model file not found. Registration skipped.")
