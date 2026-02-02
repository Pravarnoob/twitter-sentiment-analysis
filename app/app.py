import streamlit as st
import pickle
import numpy as np

# Page config
st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)

# Load models
@st.cache_resource
def load_model():
    with open("model/sentiment_pipeline.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# App UI
st.title("💬 Twitter Sentiment Analysis")
st.write("Analyze sentiment of product-related tweets using NLP")

user_input = st.text_area(
    "Enter a tweet or text:",
    height=120,
    placeholder="Example: I absolutely love the new iPhone camera!"
)

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        prediction = model.predict([user_input])[0]

        # Try probability (safe)
        try:
            probabilities = model.predict_proba([user_input])[0]
            confidence = np.max(probabilities) * 100
            st.success(f"**Sentiment:** {prediction}")
            st.info(f"**Confidence:** {confidence:.2f}%")
        except:
            st.success(f"**Sentiment:** {prediction}")

st.markdown("---")
st.caption("Built using NLP, TF-IDF & Logistic Regression")

