"""Sentiment model loading and prediction utilities."""
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "AliAmmar512/amazon-review-sentiment-distilbert"
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


def predict_sentiment(text_input: str, tokenizer, model):
    inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred_label = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][pred_label].item()
    return pred_label, confidence, probs[0].tolist()