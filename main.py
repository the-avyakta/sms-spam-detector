import joblib
from pydantic import BaseModel, field_validator
from fastapi import FastAPI, HTTPException
import numpy as np
import uvicorn

model = joblib.load('spam_detector.pkl')

app = FastAPI(
    title="Spam msg Detector",
    version="1.0",
    summary="this is a spam detector"
)

class textvalidator(BaseModel):
    textsms: str

@app.get('/health')
def health():
    return {"status": "ok"}

@app.get('/info')
def info():
    return{
        "name": "Spam Detector",
        "version": "1.0"
    }

@app.get('/')
def home():
    return{"messgae": "Welcome to SPAM SMS Detector"}

@app.post('/predict')
def predict(data: textvalidator):
    try:
        features = [ data.textsms]
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        return {
            "prediction": "Spam" if prediction==1 else "Ham",
            "probability": float(probability)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")

