import streamlit as st
import pandas as pd
import plotly.express as px
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
from sqlalchemy import create_engine, text

@st.cache_resource
def get_db_engine():
    db_url = st.secrets.get("DATABASE_URL", os.getenv("DATABASE_URL"))
    return create_engine(db_url)

engine = get_db_engine()

st.set_page_config(page_title="Amazon Review Sentiment Analytics", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_parquet("data/processed/amazon_reviews_clean.parquet")
    trend_df = pd.read_parquet("data/processed/trend_data.parquet")
    topics_df = pd.read_parquet("data/processed/negative_reviews_topics.parquet")
    return df, trend_df, topics_df

df, trend_df, topics_df = load_data()


tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview & Trends", "🤖 Model Comparison", "🗄️ SQL Explorer", "🔮 Live Prediction"])

with tab1:
    st.sidebar.header("Filters")
    selected_category = st.sidebar.selectbox(
        "Category",
        options=["All"] + sorted(df['category'].unique().tolist())
    )

    if selected_category != "All":
        filtered_trend = trend_df[trend_df['category'] == selected_category]
        filtered_topics = topics_df[topics_df['category'] == selected_category]
    else:
        filtered_trend = trend_df
        filtered_topics = topics_df

    st.subheader("Sentiment Trend Over Time (2015-2022)")
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

    st.subheader("Top Complaint Topics (Negative Reviews)")
    topic_counts = filtered_topics[filtered_topics['topic'] != -1]['topic_name'].value_counts().head(10)
    fig_topics = px.bar(
        x=topic_counts.values, y=topic_counts.index,
        orientation='h',
        labels={'x': 'Number of Reviews', 'y': ''}
    )
    fig_topics.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_topics, use_container_width=True)

with tab2:
    st.subheader("Model Comparison: TF-IDF Baseline vs Fine-tuned DistilBERT")

    comparison_data = {
        "Metric": ["Macro F1", "Accuracy", "Negative F1", "Neutral F1", "Positive F1"],
        "TF-IDF + Logistic Regression": [0.61, 0.77, 0.61, 0.34, 0.88],
        "Fine-tuned DistilBERT": [0.72, 0.89, 0.76, 0.46, 0.95]
    }
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    st.markdown("""
    **Key takeaway:** The fine-tuned transformer improved macro F1 by 11 points over the baseline,
    with the largest relative gains on the hardest classes (negative +15pts, neutral +12pts) —
    proving the heavier model earns its cost by fixing exactly where the simpler model struggled most.
    """)

    with open("data/processed/confusion_matrices.json") as f:
        cms = json.load(f)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Baseline Confusion Matrix**")
        fig1 = px.imshow(cms["baseline"],
                          labels=dict(x="Predicted", y="Actual", color="Count"),
                          x=['negative', 'neutral', 'positive'],
                          y=['negative', 'neutral', 'positive'],
                          text_auto=True, color_continuous_scale='Blues')
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown("**DistilBERT Confusion Matrix**")
        fig2 = px.imshow(cms["distilbert"],
                          labels=dict(x="Predicted", y="Actual", color="Count"),
                          x=['negative', 'neutral', 'positive'],
                          y=['negative', 'neutral', 'positive'],
                          text_auto=True, color_continuous_scale='Blues')
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    **Reading the matrices:** Notice the baseline heavily confuses neutral reviews as positive (1,396 cases) —
    the transformer cuts this error substantially (down to 315), which is the main driver of its higher neutral F1 score.
    """)

with tab3:
    st.subheader("SQL Query Explorer")
    st.write("Run live analytical queries directly against the PostgreSQL database")

    preset_queries = {
        "Average rating by category": """
            SELECT c.category_name, ROUND(AVG(r.rating), 2) as avg_rating, COUNT(*) as review_count
            FROM reviews r
            JOIN products p ON r.product_id = p.product_id
            JOIN categories c ON p.category_id = c.category_id
            GROUP BY c.category_name
            ORDER BY avg_rating DESC;
        """,
        "Sentiment breakdown by category": """
            SELECT c.category_name, r.sentiment, COUNT(*) as count
            FROM reviews r
            JOIN products p ON r.product_id = p.product_id
            JOIN categories c ON p.category_id = c.category_id
            GROUP BY c.category_name, r.sentiment
            ORDER BY c.category_name, count DESC;
        """,
        "Top 10 most helpful negative reviews": """
            SELECT r.review_text, r.helpful_vote, c.category_name
            FROM reviews r
            JOIN products p ON r.product_id = p.product_id
            JOIN categories c ON p.category_id = c.category_id
            WHERE r.sentiment = 'negative'
            ORDER BY r.helpful_vote DESC
            LIMIT 10;
        """,
        "Verified vs unverified sentiment": """
            SELECT verified_purchase, sentiment, COUNT(*) as count
            FROM reviews
            GROUP BY verified_purchase, sentiment
            ORDER BY verified_purchase, count DESC;
        """
    }

    query_choice = st.selectbox("Choose a query", list(preset_queries.keys()))
    st.code(preset_queries[query_choice], language="sql")

    if st.button("Run Query"):
        with st.spinner("Querying database..."):
            try:
                result_df = pd.read_sql(text(preset_queries[query_choice]), engine)
                st.dataframe(result_df, use_container_width=True)
                st.download_button(
                    "Download results as CSV",
                    result_df.to_csv(index=False),
                    "query_results.csv",
                    "text/csv"
                )
            except Exception as e:
                st.error(f"Query failed: {e}")

    st.divider()
    st.markdown("**Or write your own SELECT query:**")
    custom_query = st.text_area("Custom SQL (SELECT only)", height=100)
    if st.button("Run Custom Query"):
        if custom_query.strip().lower().startswith("select"):
            try:
                result_df = pd.read_sql(text(custom_query), engine)
                st.dataframe(result_df, use_container_width=True)
            except Exception as e:
                st.error(f"Query failed: {e}")
        else:
            st.warning("Only SELECT queries are allowed for safety.")


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
