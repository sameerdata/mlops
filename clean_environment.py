from azureml.core import Workspace, Environment
from azureml.exceptions import AzureMLException

# Load workspace from config.json
ws = Workspace.from_config()

target_env_name = "ml-env"

try:
    print(f"🔍 Searching for environments named '{target_env_name}'...")
    found = False
    for env in Environment.list(ws).values():
        if env.name == target_env_name:
            found = True
            print(f"🧹 Archiving environment: {env.name} (version: {env.version})")
            env._archive()
    
    if not found:
        print(f"ℹ️ No environments named '{target_env_name}' found to delete.")
    else:
        print("✅ All matching environments archived.")

except AzureMLException as e:
    print(f"⚠️ AzureML error occurred: {str(e)}")
except Exception as e:
    print(f"⚠️ Unexpected error occurred: {str(e)}")
