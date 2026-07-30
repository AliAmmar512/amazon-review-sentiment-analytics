import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

from data_loader import load_data
from db_queries import get_db_engine, is_safe_select, run_safe_query, PRESET_QUERIES, MAX_QUERY_ROWS, STATEMENT_TIMEOUT_MS
from model_utils import load_model, predict_sentiment, LABEL_MAP
from charts import sentiment_trend_chart, topic_bar_chart, confusion_matrix_chart, SENTIMENT_COLORS

st.set_page_config(page_title="Amazon Review Sentiment Analytics", layout="wide")

try:
    df, trend_df, topics_df = load_data()
except FileNotFoundError as e:
    st.error(f"Couldn't find a required data file: {e}. Check that `data/processed/` is populated.")
    st.stop()

engine = get_db_engine()

st.sidebar.header("Filters")
selected_category = st.sidebar.selectbox(
    "Category",
    options=["All"] + sorted(df["category"].unique().tolist()),
    key="category_filter",
)

st.title("📊 Amazon Review Sentiment Analytics Dashboard")
st.markdown(
    "Analysis of **74,545 Amazon reviews** across Electronics, Beauty, Home & Kitchen, and Software categories"
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Reviews", f"{len(df):,}")
col2.metric("Positive", f"{(df['sentiment'] == 'positive').mean() * 100:.1f}%")
col3.metric("Negative", f"{(df['sentiment'] == 'negative').mean() * 100:.1f}%")
col4.metric("Neutral", f"{(df['sentiment'] == 'neutral').mean() * 100:.1f}%")

tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Overview & Trends", "🤖 Model Comparison", "🗄️ SQL Explorer", "🔮 Live Prediction"]
)

# --- Tab 1: Overview & Trends ---
with tab1:
    if selected_category != "All":
        filtered_trend = trend_df[trend_df["category"] == selected_category]
        filtered_topics = topics_df[topics_df["category"] == selected_category]
    else:
        filtered_trend = trend_df
        filtered_topics = topics_df

    st.subheader("Sentiment Trend Over Time (2015-2022)")
    st.plotly_chart(sentiment_trend_chart(filtered_trend), use_container_width=True)

    st.subheader("Top Complaint Topics (Negative Reviews)")
    st.plotly_chart(topic_bar_chart(filtered_topics), use_container_width=True)

# --- Tab 2: Model Comparison ---
with tab2:
    st.subheader("Model Comparison: TF-IDF Baseline vs Fine-tuned DistilBERT")

    comparison_df = pd.DataFrame({
        "Metric": ["Macro F1", "Accuracy", "Negative F1", "Neutral F1", "Positive F1"],
        "TF-IDF + Logistic Regression": [0.61, 0.77, 0.61, 0.34, 0.88],
        "Fine-tuned DistilBERT": [0.72, 0.89, 0.76, 0.46, 0.95],
    })
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    st.markdown("""
    **Key takeaway:** The fine-tuned transformer improved macro F1 by 11 points over the baseline,
    with the largest relative gains on the hardest classes (negative +15pts, neutral +12pts) —
    proving the heavier model earns its cost by fixing exactly where the simpler model struggled most.
    """)

    cm_path = "data/processed/confusion_matrices.json"
    if os.path.exists(cm_path):
        with open(cm_path) as f:
            cms = json.load(f)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Baseline Confusion Matrix**")
            st.plotly_chart(confusion_matrix_chart(cms["baseline"]), use_container_width=True)
        with col2:
            st.markdown("**DistilBERT Confusion Matrix**")
            st.plotly_chart(confusion_matrix_chart(cms["distilbert"]), use_container_width=True)

        st.markdown("""
        **Reading the matrices:** Notice the baseline heavily confuses neutral reviews as positive (1,396 cases) —
        the transformer cuts this error substantially (down to 315), which is the main driver of its higher neutral F1 score.
        """)
    else:
        st.info("Confusion matrix data not found — skipping that section.")

# --- Tab 3: SQL Explorer ---
with tab3:
    st.subheader("SQL Query Explorer")
    st.write("Run live analytical queries directly against the PostgreSQL database")

    if engine is None:
        st.warning("No database connection configured (DATABASE_URL is not set).")
    else:
        query_choice = st.selectbox("Choose a query", list(PRESET_QUERIES.keys()), key="preset_query_choice")
        st.code(PRESET_QUERIES[query_choice], language="sql")

        if st.button("Run Query", key="run_preset_query"):
            with st.spinner("Querying database..."):
                try:
                    result_df = run_safe_query(engine, PRESET_QUERIES[query_choice], enforce_limit=True)
                    st.dataframe(result_df, use_container_width=True)
                    st.download_button("Download results as CSV", result_df.to_csv(index=False),
                                        "query_results.csv", "text/csv")
                except SQLAlchemyError as e:
                    st.error(f"Query failed: {e}")

        st.divider()
        st.markdown(f"**Or write your own SELECT query (max {MAX_QUERY_ROWS} rows, {STATEMENT_TIMEOUT_MS/1000:.0f}s timeout):**")
        custom_query = st.text_area("Custom SQL (SELECT only)", height=100, key="custom_sql")

        if st.button("Run Custom Query", key="run_custom_query"):
            ok, cleaned_or_reason = is_safe_select(custom_query)
            if not ok:
                st.warning(cleaned_or_reason)
            else:
                with st.spinner("Querying database..."):
                    try:
                        result_df = run_safe_query(engine, cleaned_or_reason, enforce_limit=True)
                        st.dataframe(result_df, use_container_width=True)
                    except SQLAlchemyError as e:
                        st.error(f"Query failed: {e}")

# --- Tab 4: Live Prediction ---
with tab4:
    st.header("🔮 Try It Yourself")
    st.write("Paste any product review below to see the model's sentiment prediction")

    user_input = st.text_area("Review text", placeholder="Type or paste a review here...", key="live_review_text")

    if st.button("Predict Sentiment", key="run_prediction"):
        if user_input.strip():
            with st.spinner("Loading model and predicting..."):
                tokenizer, model = load_model()
                pred_label_idx, confidence, probs = predict_sentiment(user_input, tokenizer, model)
                label = LABEL_MAP[pred_label_idx]

            color_map = {"negative": "🔴", "neutral": "🟡", "positive": "🟢"}
            st.subheader(f"{color_map[label]} Predicted: {label.upper()} ({confidence * 100:.1f}% confidence)")

            prob_df = pd.DataFrame({"Sentiment": ["negative", "neutral", "positive"], "Probability": probs})
            fig_prob = px.bar(prob_df, x="Sentiment", y="Probability", color="Sentiment",
                               color_discrete_map=SENTIMENT_COLORS)
            st.plotly_chart(fig_prob, use_container_width=True)
        else:
            st.warning("Please enter some text first")