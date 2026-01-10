import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="BYD EV Sentiment Analysis",
    layout="centered"
)

st.title("🚗 BYD EV Sentiment Analysis Dashboard")

st.markdown("""
This dashboard presents **consumer sentiment analysis for BYD electric vehicles**  
based on fine-tuned **BERT sentiment predictions**.
""")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_excel("data/byd_reviews.xlsx")

df = load_data()

# -----------------------------
# Sentiment Distribution
# -----------------------------
sentiment_counts = df["sentiment"].value_counts().reindex(
    ["positive", "neutral", "negative"], fill_value=0
)

# -----------------------------
# Visualization
# -----------------------------
st.subheader("📊 Sentiment Distribution")

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(
    sentiment_counts.index,
    sentiment_counts.values
)

ax.set_xlabel("Sentiment")
ax.set_ylabel("Number of Reviews")
ax.set_title("BYD Customer Sentiment Overview")

st.pyplot(fig)

# -----------------------------
# Key Metrics
# -----------------------------
st.subheader("📈 Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("😊 Positive Reviews", sentiment_counts["positive"])
col2.metric("😐 Neutral Reviews", sentiment_counts["neutral"])
col3.metric("😡 Negative Reviews", sentiment_counts["negative"])

# -----------------------------
# Overall Sentiment Conclusion
# -----------------------------
st.subheader("🧠 Overall Sentiment Conclusion")

total_reviews = sentiment_counts.sum()
positive_ratio = sentiment_counts["positive"] / total_reviews * 100

if positive_ratio >= 60:
    conclusion = (
        "Overall consumer sentiment towards **BYD EVs is strongly positive**. "
        "Most users express satisfaction with the vehicle performance, features, "
        "and overall ownership experience."
    )
elif positive_ratio >= 40:
    conclusion = (
        "Consumer sentiment towards **BYD EVs is mixed**. "
        "While many users are satisfied, there are notable neutral and negative opinions "
        "that highlight areas for improvement."
    )
else:
    conclusion = (
        "Overall sentiment towards **BYD EVs is predominantly negative**, "
        "indicating potential dissatisfaction among consumers."
    )

st.success(conclusion)

# -----------------------------
# Raw Data Preview
# -----------------------------
with st.expander("📂 View Sample Reviews"):
    st.dataframe(df.head(10))
