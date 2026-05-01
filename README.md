# Twitter Sentiment Analysis for Product Feedback

This project implements an end-to-end Twitter sentiment analysis system using Natural Language Processing (NLP).  
It includes data exploration, preprocessing, TF-IDF feature extraction, model training with Logistic Regression, 
and deployment using Streamlit for real-time sentiment prediction.

## Features
- Exploratory Data Analysis (EDA)
- Text preprocessing and cleaning
- TF-IDF based feature extraction
- Multi-class sentiment classification
- End-to-end ML pipeline
- Streamlit-based deployment

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- NLTK
- Streamlit

## Status
Model trained and pipeline created. Deployment in progress.

## 🚀 Live Demo

🔗 **Deployed Application:**  

https://twitter-sentiment-analysis-6-xkmh.onrender.com/

---

## Sentiment Analysis API

This project includes a standalone REST API built using FastAPI to serve sentiment predictions for text and CSV inputs.

🚀 **Live API**

Base URL:
https://twitter-sentiment-analysis-7-7mgm.onrender.com/

Interactive API Docs:
https://twitter-sentiment-analysis-7-7mgm.onrender.com/docs

**Features**

-  Single text sentiment prediction
-  Batch prediction using CSV upload
-  Confidence score output
-  Deployed and publicly accessible


**API Endpoints**

1. Predict Sentiment (Single Text)

POST "/predict"

Request Body:

{
  "text": "I love this product"
}

Response:

{
  "input": "I love this product",
  "sentiment": "positive",
  "confidence": 97.5
}


2. Batch Prediction (CSV Upload)

POST "/predict_csv"

Input:

Upload a CSV file with a column named "text"

Example:

text
I love this product
This is terrible

Response:

{
  "total_rows": 2,
  "results": [
    {"text": "I love this product", "sentiment": "positive"},
    {"text": "This is terrible", "sentiment": "negative"}
  ]
}


**Example Usage (Python)**

import requests

url = "https://twitter-sentiment-analysis-7-7mgm.onrender.com/predict"

data = {"text": "Amazing product!"}

response = requests.post(url, json=data)
print(response.json())


**Architecture**

The API exposes the trained sentiment analysis model via REST endpoints, enabling integration with frontend applications or external services.


⚠️ **Note**

- The API is hosted on Render (free tier), so the first request may take a few seconds due to cold start.
