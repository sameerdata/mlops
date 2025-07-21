from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment, Dataset
from azureml.core.compute import ComputeTarget, AmlCompute
import os

# Load workspace
ws = Workspace.from_config()

# Load dataset
dataset = Dataset.get_by_name(ws, 'customer_churn_dataset')

# Create/attach compute cluster
cluster_name = "cpu-cluster"
if cluster_name in ws.compute_targets:
    compute_target = ws.compute_targets[cluster_name]
else:
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_DS11_V2',
                                                           max_nodes=2)
    compute_target = ComputeTarget.create(ws, cluster_name, compute_config)
    compute_target.wait_for_completion(show_output=True)

# Define environment
env = Environment.from_pip_requirements(name='ml-env', file_path='src/requirements.txt')

# Download dataset to compute target and pass as argument
df = dataset.to_pandas_dataframe()
csv_path = os.path.join('src', 'customer_churn_100.csv')
df.to_csv(csv_path, index=False)


# Script run config
src = ScriptRunConfig(
    source_directory='src',
    script='train.py',
    arguments=['--data_path', 'customer_churn_100.csv'],  # just the filename
    compute_target=compute_target,
    environment=env
)

# Submit experiment
experiment = Experiment(workspace=ws, name='churn-training')
run = experiment.submit(src)
run.wait_for_completion(show_output=True)
