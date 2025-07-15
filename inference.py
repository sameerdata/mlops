import joblib
import os
import json

# Load model
def model_fn(model_dir):
    model_path = os.path.join(model_dir, "sklearn_model.pkl")
    return joblib.load(model_path)

# Parse input
def input_fn(request_body, content_type):
    if content_type == "application/json":
        return json.loads(request_body)["instances"]
    raise ValueError("Unsupported content type: " + content_type)

# Make prediction
def predict_fn(input_data, model):
    return model.predict(input_data).tolist()
