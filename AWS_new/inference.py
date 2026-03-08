import pickle
import numpy as np
import json

def model_fn(model_dir):
    with open(model_dir + "/xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

def input_fn(request_body, request_content_type):
    data = json.loads(request_body)
    return np.array(data).reshape(1,-1)

def predict_fn(input_data, model):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    return {
        "sepsis_prediction": int(prediction[0]),
        "risk_probability": float(probability)
    }