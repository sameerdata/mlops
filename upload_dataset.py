import argparse
from azureml.core import Workspace, Datastore, Dataset

parser = argparse.ArgumentParser()
parser.add_argument("--resource_group", type=str, required=True)
parser.add_argument("--workspace_name", type=str, required=True)
parser.add_argument("--region", type=str, required=True)
parser.add_argument("--dataset", type=str, required=True)

args = parser.parse_args()

ws = Workspace.get(
    name=args.workspace_name,
    resource_group=args.resource_group,
)

datastore = ws.get_default_datastore()

# Upload dataset to datastore
datastore.upload(
    src_dir='dataset',
    target_path='datasets/',
    overwrite=True,
    show_progress=True
)

# Register as tabular dataset
dataset = Dataset.Tabular.from_delimited_files(path=(datastore, f'datasets/{args.dataset}'))
dataset = dataset.register(workspace=ws, name='customer_churn_data', create_new_version=True)

print("✅ Dataset uploaded and registered.")
