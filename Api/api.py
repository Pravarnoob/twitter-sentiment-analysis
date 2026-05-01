from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import pandas as pd
import pickle
import numpy as np

app = FastAPI(title="Sentiment Analysis API")

# Load model
with open("model/sentiment_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

# -------- INPUT SCHEMA --------
class TextInput(BaseModel):
    text: str

# -------- ROOT --------
@app.get("/")
def home():
    return {"message": "Sentiment API is running"}

# -------- SINGLE TEXT --------
@app.post("/predict")
def predict(data: TextInput):
    pred = model.predict([data.text])[0]

    try:
        prob = model.predict_proba([data.text])[0]
        confidence = float(np.max(prob) * 100)
    except:
        confidence = None

    return {
        "input": data.text,
        "sentiment": pred,
        "confidence": confidence
    }

# -------- CSV --------
@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)

    if "text" not in df.columns:
        return {"error": "CSV must contain 'text' column"}

    preds = model.predict(df["text"].tolist())
    df["sentiment"] = preds

    return {
        "total_rows": len(df),
        "results": df.to_dict(orient="records")
    }
