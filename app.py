import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Amazon Review Sentiment Analytics", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_parquet("data/processed/amazon_reviews_clean.parquet")
    trend_df = pd.read_parquet("data/processed/trend_data.parquet")
    topics_df = pd.read_parquet("data/processed/negative_reviews_topics.parquet")
    return df, trend_df, topics_df

df, trend_df, topics_df = load_data()

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

@st.cache_resource
def load_model():
    model_name = "AliAmmar512/amazon-review-sentiment-distilbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()
label_map = {0: "negative", 1: "neutral", 2: "positive"}

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred_label = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][pred_label].item()
    return label_map[pred_label], confidence, probs[0].tolist()
st.header("🔮 Try It Yourself")
st.write("Paste any product review below to see the model's sentiment prediction")

user_input = st.text_area("Review text", placeholder="Type or paste a review here...")

if st.button("Predict Sentiment"):
    if user_input.strip():
        label, confidence, probs = predict_sentiment(user_input)

        color_map = {"negative": "🔴", "neutral": "🟡", "positive": "🟢"}
        st.subheader(f"{color_map[label]} Predicted: {label.upper()} ({confidence*100:.1f}% confidence)")

        prob_df = pd.DataFrame({
            'Sentiment': ['negative', 'neutral', 'positive'],
            'Probability': probs
        })
        fig_prob = px.bar(prob_df, x='Sentiment', y='Probability', color='Sentiment',
                           color_discrete_map={'negative': '#e74c3c', 'neutral': '#f39c12', 'positive': '#2ecc71'})
        st.plotly_chart(fig_prob, use_container_width=True)
    else:
        st.warning("Please enter some text first")
st.title("📊 Amazon Review Sentiment Analytics Dashboard")

st.markdown("Analysis of **74,545 Amazon reviews** across Electronics, Beauty, Home & Kitchen, and Software categories")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Reviews", f"{len(df):,}")
col2.metric("Positive", f"{(df['sentiment']=='positive').mean()*100:.1f}%")
col3.metric("Negative", f"{(df['sentiment']=='negative').mean()*100:.1f}%")
col4.metric("Neutral", f"{(df['sentiment']=='neutral').mean()*100:.1f}%")

st.sidebar.header("Filters")
selected_category = st.sidebar.selectbox(
    "Category",
    options=["All"] + sorted(df['category'].unique().tolist())
)

if selected_category != "All":
    filtered_trend = trend_df[trend_df['category'] == selected_category]
else:
    filtered_trend = trend_df

st.header("Sentiment Trend Over Time (2015-2022)")

sentiment_by_year = filtered_trend.groupby(['review_year', 'sentiment']).size().reset_index(name='count')
totals_by_year = filtered_trend.groupby('review_year').size().reset_index(name='total')
sentiment_by_year = sentiment_by_year.merge(totals_by_year, on='review_year')
sentiment_by_year['pct'] = (sentiment_by_year['count'] / sentiment_by_year['total']) * 100

fig_trend = px.line(
    sentiment_by_year, x='review_year', y='pct', color='sentiment',
    markers=True,
    color_discrete_map={'negative': '#e74c3c', 'neutral': '#f39c12', 'positive': '#2ecc71'},
    labels={'pct': '% of Reviews', 'review_year': 'Year'}
)
st.plotly_chart(fig_trend, use_container_width=True)
st.header("Top Complaint Topics (Negative Reviews)")

if selected_category != "All":
    filtered_topics = topics_df[topics_df['category'] == selected_category]
else:
    filtered_topics = topics_df

topic_counts = filtered_topics[filtered_topics['topic'] != -1]['topic_name'].value_counts().head(10)

fig_topics = px.bar(
    x=topic_counts.values, y=topic_counts.index,
    orientation='h',
    labels={'x': 'Number of Reviews', 'y': ''}
)
fig_topics.update_layout(yaxis={'categoryorder': 'total ascending'})
st.plotly_chart(fig_topics, use_container_width=True)
