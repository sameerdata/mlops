import argparse
from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment

parser = argparse.ArgumentParser()
parser.add_argument("--resource_group", type=str, required=True)
parser.add_argument("--workspace_name", type=str, required=True)
parser.add_argument("--region", type=str, required=True)

args = parser.parse_args()

ws = Workspace.get(
    name=args.workspace_name,
    resource_group=args.resource_group,
)

# Set up Azure ML environment
env = Environment.from_conda_specification(
    name="sklearn-env",
    file_path="conda.yml"  # You can also inline the packages
)

# Create script config (assumes train.py exists)
src = ScriptRunConfig(
    source_directory='.',
    script='train.py',
    arguments=[],
    environment=env,
    compute_target="cpu-cluster"  # Assumes you've already created a cluster
)

exp = Experiment(workspace=ws, name="churn-training")
run = exp.submit(config=src)
run.wait_for_completion(show_output=True)
