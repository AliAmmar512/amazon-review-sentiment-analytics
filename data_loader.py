"""Data loading utilities for the Amazon Review Sentiment Analytics dashboard."""
import streamlit as st
import pandas as pd


@st.cache_data
def load_data():
    df = pd.read_parquet("data/processed/amazon_reviews_clean.parquet")
    trend_df = pd.read_parquet("data/processed/trend_data.parquet")
    topics_df = pd.read_parquet("data/processed/negative_reviews_topics.parquet")
    return df, trend_df, topics_df