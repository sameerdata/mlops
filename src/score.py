import joblib
import json
import numpy as np
import os
from azureml.core.model import Model

def init():
    global model
    model_path = Model.get_model_path(model_name='churn_model')
    model = joblib.load(model_path)

def run(raw_data):
    try:
        data = json.loads(raw_data)
        input_data = data.get("data")
        if input_data is None:
            return {'error': 'Missing "data" in request.'}

        input_array = np.array(input_data)
        prediction = model.predict(input_array).tolist()
        return {'prediction': prediction}
    except Exception as e:
        return {'error': f'Exception during inference: {str(e)}'}
