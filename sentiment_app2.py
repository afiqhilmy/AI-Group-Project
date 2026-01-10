import streamlit as st
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os
from transformers import BertTokenizer, BertForSequenceClassification

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="EV Sentiment Comparison",
    layout="wide"
)

st.title("🚗 EV Brand Sentiment Analysis Comparison")
st.markdown("""
This application compares **consumer sentiment** between **BYD** and **Proton EMAS** EV brands  
using fine-tuned **BERT sentiment analysis models**.
""")

# ---------------------------
# Resolve Base Directory
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BYD_MODEL_PATH = os.path.join(BASE_DIR, "models", "byd")
EMAS_MODEL_PATH = os.path.join(BASE_DIR, "models", "emas")

BYD_DATA_PATH = os.path.join(BASE_DIR, "data", "byd_reviews.xlsx")
EMAS_DATA_PATH = os.path.join(BASE_DIR, "data", "emas_reviews.xlsx")

# ---------------------------
# Load Models & Tokenizers
# ---------------------------
@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"❌ Model folder not found: {model_path}")
        st.stop()

    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

byd_tokenizer, byd_model = load_model(BYD_MODEL_PATH)
emas_tokenizer, emas_model = load_model(EMAS_MODEL_PATH)

# ---------------------------
# Load Data
# ---------------------------
@st.cache_data
def load_data(path):
    if not os.path.exists(path):
        st.error(f"❌ Data file not found: {path}")
        st.stop()
    return pd.read_excel(path)

byd_df = load_data(BYD_DATA_PATH)
emas_df = load_data(EMAS_DATA_PATH)

# ---------------------------
# Sentiment Prediction Function
# ---------------------------
label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

def predict_sentiment(texts, tokenizer, model):
    sentiments = []
    for text in texts:
        inputs = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()

        sentiments.append(label_map[pred])
    return sentiments

# ---------------------------
# Run Predictions
# ---------------------------
byd_df["Predicted_Sentiment"] = predict_sentiment(
    byd_df["review_text"].astype(str),
    byd_tokenizer,
    byd_model
)

emas_df["Predicted_Sentiment"] = predict_sentiment(
    emas_df["review_text"].astype(str),
    emas_tokenizer,
    emas_model
)

# ---------------------------
# Aggregate Results
# ---------------------------
byd_counts = byd_df["Predicted_Sentiment"].value_counts().reindex(
    ["Positive", "Neutral", "Negative"], fill_value=0
)

emas_counts = emas_df["Predicted_Sentiment"].value_counts().reindex(
    ["Positive", "Neutral", "Negative"], fill_value=0
)

# ---------------------------
# Visualization
# ---------------------------
st.subheader("📊 Sentiment Distribution Comparison")

fig, ax = plt.subplots(figsize=(8, 5))
x = range(3)

ax.bar(x, byd_counts, width=0.4, label="BYD", align="center")
ax.bar(x, emas_counts, width=0.4, label="EMAS", align="edge")

ax.set_xticks(x)
ax.set_xticklabels(["Positive", "Neutral", "Negative"])
ax.set_ylabel("Number of Reviews")
ax.set_title("Sentiment Comparison Between EV Brands")
ax.legend()

st.pyplot(fig)

# ---------------------------
# Metrics Section
# ---------------------------
col1, col2 = st.columns(2)

with col1:
    st.metric("BYD Positive Reviews", byd_counts["Positive"])
    st.metric("BYD Negative Reviews", byd_counts["Negative"])

with col2:
    st.metric("EMAS Positive Reviews", emas_counts["Positive"])
    st.metric("EMAS Negative Reviews", emas_counts["Negative"])

# ---------------------------
# Best EV Brand Decision
# ---------------------------
st.subheader("🏆 Best EV Brand Based on Sentiment")

byd_score = byd_counts["Positive"] - byd_counts["Negative"]
emas_score = emas_counts["Positive"] - emas_counts["Negative"]

if byd_score > emas_score:
    best_brand = "BYD"
    justification = (
        "BYD shows a higher proportion of **positive sentiment** and fewer negative reviews, "
        "indicating stronger customer satisfaction."
    )
else:
    best_brand = "Proton EMAS"
    justification = (
        "Proton EMAS receives comparatively more **positive sentiment**, "
        "suggesting better overall consumer perception."
    )

st.success(f"**Best EV Brand:** {best_brand}")
st.write("**Justification:**")
st.write(justification)

# ---------------------------
# Raw Data (Optional)
# ---------------------------
with st.expander("📂 View Raw Predictions"):
    st.write("BYD Predictions")
    st.dataframe(byd_df.head())

    st.write("EMAS Predictions")
    st.dataframe(emas_df.head())
