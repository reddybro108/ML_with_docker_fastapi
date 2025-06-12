from fastapi import FastAPI
from app.schema import ClientData
from app.model import predict

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Bank Marketing ML API is running."}


@app.post("/predict")
def make_prediction(data: ClientData):
    result = predict(data)
    return {"prediction": "yes" if result == 1 else "no"}
