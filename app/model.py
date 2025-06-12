import pickle
import numpy as np
from app.schema import ClientData

# Load model and encoders
with open("app/bank_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("app/encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

def predict(data: ClientData):
    raw = data.dict()
    for col, encoder in encoders.items():
        raw[col] = encoder.transform([raw[col]])[0]
    input_data = np.array([list(raw.values())])
    prediction = model.predict(input_data)[0]
    return int(prediction)
