import boto3
import time
import os

# Environment from Jenkins
bucket = os.environ['BUCKET']
model_file = 'model.tar.gz'
model_name = f"mlops-model-{int(time.time())}"
endpoint_config_name = f"{model_name}-config"
endpoint_name = f"{model_name}-endpoint"
role = os.environ['SAGEMAKER_ROLE']
region = os.environ['AWS_DEFAULT_REGION']

s3_path = f"s3://{bucket}/models/{model_file}"

# ✅ Correct built-in Scikit-learn container
container = {
    'Image': '683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3',
    'ModelDataUrl': s3_path,
    'Environment': {
        'SAGEMAKER_PROGRAM': 'inference.py',  # ✅ must match archive root structure
        'SAGEMAKER_SUBMIT_DIRECTORY': s3_path
    }
}

# SageMaker client
sagemaker = boto3.client('sagemaker', region_name=region)

# 1. Create Model
print("🚀 Creating SageMaker model...")
sagemaker.create_model(
    ModelName=model_name,
    ExecutionRoleArn=role,
    PrimaryContainer=container
)

# 2. Create Endpoint Config
print("⚙️ Creating endpoint configuration...")
sagemaker.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[{
        'VariantName': 'AllTraffic',
        'ModelName': model_name,
        'InitialInstanceCount': 1,
        'InstanceType': 'ml.t2.medium'
    }],
    DataCaptureConfig={
        'EnableCapture': True,
        'InitialSamplingPercentage': 100,
        'DestinationS3Uri': f's3://{bucket}/logs/',
        'CaptureOptions': [{'CaptureMode': 'Input'}, {'CaptureMode': 'Output'}],
        'CaptureContentTypeHeader': {
            'JsonContentTypes': ['application/json']
        }
    }
)

# 3. Deploy Endpoint
print(f"📡 Deploying endpoint: {endpoint_name} ...")
sagemaker.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=endpoint_config_name
)

print("✅ Deployment initiated.")
print(f"➡️ View status: https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/endpoints/{endpoint_name}")
