import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)

# Load model
@st.cache_resource
def load_model():
    with open("model/sentiment_pipeline.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# Title
st.title("💬 Twitter Sentiment Analysis")
st.write("Analyze sentiment of product-related tweets using NLP")

# ---------------- SINGLE INPUT ----------------
st.subheader("🔹 Single Text Prediction")

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

        try:
            probabilities = model.predict_proba([user_input])[0]
            confidence = np.max(probabilities) * 100
            st.success(f"**Sentiment:** {prediction}")
            st.info(f"**Confidence:** {confidence:.2f}%")
        except:
            st.success(f"**Sentiment:** {prediction}")

# ---------------- CSV INPUT ----------------
st.markdown("---")
st.subheader("📂 Batch Prediction (CSV Upload)")

uploaded_file = st.file_uploader("Upload CSV file (must contain 'text' column)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "text" not in df.columns:
        st.error("CSV must contain a column named 'text'")
    else:
        st.write("📊 Uploaded Data:")
        st.dataframe(df.head())

        # Predict
        predictions = model.predict(df["text"].tolist())
        df["sentiment"] = predictions

        st.write("✅ Predictions:")
        st.dataframe(df)

        # ---------------- PIE CHART ----------------
        st.subheader("📊 Sentiment Distribution")

        sentiment_counts = df["sentiment"].value_counts()

        fig, ax = plt.subplots()
        ax.pie(
            sentiment_counts,
            labels=sentiment_counts.index,
            autopct="%1.1f%%"
        )
        ax.set_title("Sentiment Distribution")

        st.pyplot(fig)

        # ---------------- DOWNLOAD ----------------
        st.download_button(
            "⬇️ Download Results",
            df.to_csv(index=False),
            "sentiment_results.csv",
            "text/csv"
        )

# Footer
st.markdown("---")
st.caption("Built using NLP, TF-IDF & Logistic Regression")
