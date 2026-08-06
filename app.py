import json
import os
import time
import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

from data_loader import load_data
from db_queries import get_db_engine, is_safe_select, run_safe_query, PRESET_QUERIES, MAX_QUERY_ROWS, STATEMENT_TIMEOUT_MS
from model_utils import load_model, predict_sentiment, LABEL_MAP
from charts import sentiment_trend_chart, topic_bar_chart, confusion_matrix_chart, SENTIMENT_COLORS



st.set_page_config(page_title="Amazon Review Sentiment Analytics", layout="wide")



def animated_metric(col, label, target_value, is_percent=False, duration=0.5, key=None):
    """Animate a metric counting up to its value, but only once per session."""
    session_key = f"animated_{key}"
    placeholder = col.empty()

    if st.session_state.get(session_key, False):
        display_val = f"{target_value:.1f}%" if is_percent else f"{int(target_value):,}"
        placeholder.metric(label, display_val)
        return

    steps = 20
    for i in range(steps + 1):
        current = target_value * (i / steps)
        display_val = f"{current:.1f}%" if is_percent else f"{int(current):,}"
        placeholder.metric(label, display_val)
        time.sleep(duration / steps)

    st.session_state[session_key] = True
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&family=Inter:wght@400;500&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        animation: fadeIn 0.7s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(14px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Animated glow background orbs */
    .stApp {
        background:
            radial-gradient(circle at 15% 20%, rgba(108, 92, 231, 0.12) 0%, transparent 40%),
            radial-gradient(circle at 85% 80%, rgba(162, 155, 254, 0.10) 0%, transparent 40%),
            #0E1117;
    }

    /* Glassmorphism metric cards */
    div[data-testid="stMetric"] {
        background: rgba(26, 29, 41, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(108, 92, 231, 0.25);
        border-radius: 16px;
        padding: 18px;
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-6px) scale(1.02);
        box-shadow: 0 12px 32px rgba(108, 92, 231, 0.3);
        border-color: rgba(108, 92, 231, 0.6);
    }

    /* Headings use Poppins for more character */
    h1, h2, h3 {
        font-family: 'Poppins', sans-serif !important;
    }

    /* Buttons with glow-on-hover */
    .stButton > button {
        background: linear-gradient(135deg, #6C5CE7 0%, #A29BFE 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 1.6rem;
        font-weight: 600;
        font-family: 'Poppins', sans-serif;
        transition: all 0.25s ease;
    }
    .stButton > button:hover {
        transform: scale(1.04);
        box-shadow: 0 0 24px rgba(108, 92, 231, 0.55);
    }
    .stButton > button:active {
        transform: scale(0.98);
    }

    /* Tabs with animated underline */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        font-weight: 600;
        font-family: 'Poppins', sans-serif;
        transition: background 0.2s ease;
        padding: 10px 18px;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(108, 92, 231, 0.1);
    }
    .stTabs [aria-selected="true"] {
        background: rgba(108, 92, 231, 0.15) !important;
        border-bottom: 3px solid #6C5CE7 !important;
    }

    /* DataFrames / tables with rounded corners */
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(108, 92, 231, 0.15);
    }

    /* Text area / input glow on focus */
    .stTextArea textarea, .stTextInput input {
        border-radius: 10px !important;
        transition: box-shadow 0.2s ease, border-color 0.2s ease;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        box-shadow: 0 0 0 3px rgba(108, 92, 231, 0.3) !important;
        border-color: #6C5CE7 !important;
    }

    /* Sidebar polish */
    section[data-testid="stSidebar"] {
        background: rgba(15, 17, 25, 0.95);
        border-right: 1px solid rgba(108, 92, 231, 0.15);
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #0E1117; }
    ::-webkit-scrollbar-thumb {
        background: rgba(108, 92, 231, 0.5);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover { background: rgba(108, 92, 231, 0.8); }

    /* Hide default Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background: transparent;}
</style>
""", unsafe_allow_html=True)
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
st.markdown("""
<div style="text-align: center; padding: 2rem 0 1rem 0;">
    <h1 style="font-size: 2.8rem; margin-bottom: 0.3rem;
               background: linear-gradient(135deg, #6C5CE7, #A29BFE);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        📊 Amazon Review Sentiment Analytics
    </h1>
    <p style="color: #9CA3AF; font-size: 1.1rem;">
        Sentiment analysis, topic modeling & live prediction across 74,545 real Amazon reviews
    </p>
</div>
""", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
animated_metric(col1, "Total Reviews", len(df), key="total")
animated_metric(col2, "Positive", (df['sentiment'] == 'positive').mean() * 100, is_percent=True, key="positive")
animated_metric(col3, "Negative", (df['sentiment'] == 'negative').mean() * 100, is_percent=True, key="negative")
animated_metric(col4, "Neutral", (df['sentiment'] == 'neutral').mean() * 100, is_percent=True, key="neutral")
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