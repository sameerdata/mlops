import boto3
import time
import os

bucket = os.environ['BUCKET']
model_file = 'model.pkl'
model_name = f"mlops-model-{int(time.time())}"
endpoint_config_name = f"{model_name}-config"
endpoint_name = f"{model_name}-endpoint"
role = os.environ['SAGEMAKER_ROLE']
region = os.environ['AWS_DEFAULT_REGION']

s3_path = f"s3://{bucket}/models/{model_file}"
container = {
    'Image': '683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:0.20.0-cpu-py3',
    'ModelDataUrl': s3_path
}

sagemaker = boto3.client('sagemaker', region_name=region)

# 1. Create Model
print("Creating SageMaker model...")
sagemaker.create_model(
    ModelName=model_name,
    ExecutionRoleArn=role,
    PrimaryContainer=container
)

# 2. Create Endpoint Config
print("Creating endpoint config...")
sagemaker.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[
        {
            'VariantName': 'AllTraffic',
            'ModelName': model_name,
            'InitialInstanceCount': 1,
            'InstanceType': 'ml.t2.medium'
        }
    ]
)

# 3. Deploy Endpoint
print(f"Deploying endpoint {endpoint_name}...")
sagemaker.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=endpoint_config_name
)

print("✅ Deployment initiated. Use AWS console to monitor status.")
