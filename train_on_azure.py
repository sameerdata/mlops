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

# Convert dataset to DataFrame and save as CSV
print("📦 Converting Azure ML dataset to CSV file...")
df = dataset.to_pandas_dataframe()
csv_path = os.path.join('src', 'customer_churn_100.csv')
df.to_csv(csv_path, index=False)

# Configure the training job
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

# Register the trained model
print("📦 Registering trained model...")
model_path = os.path.join('src', 'sklearn_model.pkl')  # path should match train.py output
if os.path.exists(model_path):
    registered_model = Model.register(
        workspace=ws,
        model_path=model_path,
        model_name="churn_model"
    )
    print(f"✅ Model registered: {registered_model.name} (v{registered_model.version})")
else:
    print("❌ Model file not found. Registration skipped.")
