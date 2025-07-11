import boto3
import pandas as pd
import numpy as np
import argparse
import os
import time
from sagemaker import Session, get_execution_role
from sagemaker.inputs import TrainingInput
from sagemaker.estimator import Estimator
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer

# ------------------------- 🔧 ARGUMENT PARSING -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--bucket", required=True)
parser.add_argument("--data_file", required=True)
parser.add_argument("--region", required=True)
parser.add_argument("--role", required=True)
args = parser.parse_args()

bucket = args.bucket
data_file = args.data_file
region = args.region
role = args.role
prefix = "xgboost-churn"

# ------------------------- ☁️ SESSION SETUP -------------------------
session = Session()
s3 = boto3.client("s3", region_name=region)

# ------------------------- 📥 LOAD AND SPLIT DATA -------------------------
df = pd.read_csv(data_file)

if "churn" not in df.columns:
    raise ValueError("❌ 'churn' column missing in dataset.")

X = df.drop("churn", axis=1)
y = df["churn"]

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

train_df = pd.DataFrame(X_train_scaled)
train_df["churn"] = y_train.reset_index(drop=True)
test_df = pd.DataFrame(X_test_scaled)
test_df["churn"] = y_test.reset_index(drop=True)

train_df.to_csv("train.csv", index=False, header=False)
test_df.to_csv("test.csv", index=False, header=False)

# ------------------------- 📤 UPLOAD TO S3 -------------------------
s3.upload_file("train.csv", bucket, f"{prefix}/train.csv")
s3.upload_file("test.csv", bucket, f"{prefix}/test.csv")

# ------------------------- 🧠 XGBOOST CONFIG -------------------------
container = Estimator(
    image_uri=Session().image_uri("xgboost", region=region, version="1.3-1"),
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    output_path=f"s3://{bucket}/{prefix}/output",
    sagemaker_session=session,
    hyperparameters={
        "max_depth": 5,
        "eta": 0.2,
        "objective": "binary:logistic",
        "num_round": 100
    }
)

container.fit({"train": TrainingInput(f"s3://{bucket}/{prefix}/train.csv", content_type="text/csv")})

# ------------------------- 🚀 DEPLOY MODEL -------------------------
predictor = container.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name="xgboost-churn-endpoint"
)

# ------------------------- 🔮 PREDICT -------------------------
time.sleep(60)  # wait for endpoint to be fully up

predictor.serializer = CSVSerializer()
predictor.deserializer = JSONDeserializer()

# Read S3 test.csv to make a prediction
test = pd.read_csv("test.csv", header=None)
test = test[~test.apply(lambda row: row.astype(str).str.contains("churn", case=False).any(), axis=1)]
sample = test.iloc[0, :-1].to_numpy().astype(float)
csv_input = ",".join(map(str, sample))

prediction = predictor.predict(csv_input)
print("🔮 Predicted probability of churn:", prediction)

