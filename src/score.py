import joblib
import json
import numpy as np
from azureml.core.model import Model

def init():
    global model
    model_path = Model.get_model_path('churn_model')
    model = joblib.load(model_path)

def run(data):
    try:
        input_data = json.loads(data)['data']
        input_array = np.array(input_data)
        result = model.predict(input_array).tolist()
        return {'prediction': result}
    except Exception as e:
        return {'error': str(e)}
