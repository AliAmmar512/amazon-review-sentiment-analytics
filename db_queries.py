"""Database connection and query utilities for the SQL Explorer tab.

NOTE: for production, DATABASE_URL should point at a role that only has
SELECT privileges on the reviews/products/categories tables. The query
guardrails below reduce risk but are not a substitute for DB-level
permissions - never rely on app-layer checks alone for a public-facing DB.
"""
import re
import os

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

MAX_QUERY_ROWS = 1000
STATEMENT_TIMEOUT_MS = 5000  # kill runaway queries after 5s

# Keywords that have no business showing up in a read-only explorer.
_BLOCKED_KEYWORDS = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|"
    r"exec|execute|call|copy|merge|attach|vacuum|do)\b",
    re.IGNORECASE,
)

PRESET_QUERIES = {
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
    """,
}


@st.cache_resource
def get_db_engine():
    db_url = st.secrets.get("DATABASE_URL", os.getenv("DATABASE_URL"))
    if not db_url:
        return None
    return create_engine(db_url, pool_pre_ping=True)


def is_safe_select(query: str) -> tuple[bool, str]:
    """Best-effort guard for the free-text SQL box.

    Rejects anything that isn't a single, plain SELECT statement.
    """
    stripped = query.strip().rstrip(";").strip()
    if not stripped:
        return False, "Query is empty."
    if ";" in stripped:
        return False, "Multiple statements aren't allowed."
    if not stripped.lower().startswith("select"):
        return False, "Only SELECT queries are allowed."
    if _BLOCKED_KEYWORDS.search(stripped):
        return False, "Query contains a disallowed keyword."
    return True, stripped


def run_safe_query(engine, query: str, enforce_limit: bool = False) -> pd.DataFrame:
    """Run a SELECT with a statement timeout and a hard row cap."""
    q = query
    if enforce_limit and "limit" not in q.lower():
        q = f"{q}\nLIMIT {MAX_QUERY_ROWS}"

    with engine.connect() as conn:
        conn.execute(text(f"SET LOCAL statement_timeout = {STATEMENT_TIMEOUT_MS}"))
        result = conn.execute(text(q))
        rows = result.fetchmany(MAX_QUERY_ROWS)
        cols = result.keys()
        return pd.DataFrame(rows, columns=cols)