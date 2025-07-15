import boto3
import time
import os

# Load environment variables from Jenkins
bucket = os.environ['BUCKET']
model_file = 'model.pkl'
model_name = f"mlops-model-{int(time.time())}"
endpoint_config_name = f"{model_name}-config"
endpoint_name = f"{model_name}-endpoint"
role = os.environ['SAGEMAKER_ROLE']
region = os.environ['AWS_DEFAULT_REGION']

# S3 model path
s3_path = f"s3://{bucket}/models/{model_file}"

# Use scikit-learn prebuilt container
container = {
    'Image': '683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:0.20.0-cpu-py3',
    'ModelDataUrl': s3_path,
    'Environment': {
        'SAGEMAKER_REGION': region
    }
}

# Initialize SageMaker client
sagemaker = boto3.client('sagemaker', region_name=region)

# 1. Create Model
print("🚀 Creating SageMaker model...")
sagemaker.create_model(
    ModelName=model_name,
    ExecutionRoleArn=role,
    PrimaryContainer=container
)

# 2. Create Endpoint Configuration with Data Capture + CloudWatch
print("🛠️ Creating endpoint configuration with logging and data capture...")
sagemaker.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[
        {
            'VariantName': 'AllTraffic',
            'ModelName': model_name,
            'InitialInstanceCount': 1,
            'InstanceType': 'ml.t2.medium'
        }
    ],
    DataCaptureConfig={
        'EnableCapture': True,
        'InitialSamplingPercentage': 100,
        'DestinationS3Uri': f's3://{bucket}/logs/',
        'CaptureOptions': [
            {'CaptureMode': 'Input'},
            {'CaptureMode': 'Output'}
        ],
        'CaptureContentTypeHeader': {
            'JsonContentTypes': ['application/json']
        }
    }
)

# 3. Create Endpoint
print(f"📡 Deploying endpoint: {endpoint_name} ...")
sagemaker.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=endpoint_config_name
)

print("✅ Deployment initiated.")
print(f"➡️  Monitor here: https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/endpoints/{endpoint_name}")
