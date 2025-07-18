from azureml.core import Workspace, Dataset
import os

# Get workspace from config or environment
ws = Workspace.from_config()  # assumes config.json OR use arguments if dynamic

# Dataset path
data_path = os.path.join("dataset", "customer_churn_100.csv")

# Upload as dataset
datastore = ws.get_default_datastore()
datastore.upload_files(files=[data_path],
                       target_path='datasets/',
                       overwrite=True)

# Register dataset
dataset = Dataset.Tabular.from_delimited_files(path=(datastore, 'datasets/customer_churn_100.csv'))
dataset = dataset.register(workspace=ws,
                           name='customer_churn_dataset',
                           create_new_version=True)

print("✅ Dataset uploaded and registered.")
