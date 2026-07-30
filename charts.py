"""Chart-building functions for the dashboard."""
import pandas as pd
import plotly.express as px

SENTIMENT_COLORS = {"negative": "#e74c3c", "neutral": "#f39c12", "positive": "#2ecc71"}


def sentiment_trend_chart(trend_df: pd.DataFrame):
    sentiment_by_year = trend_df.groupby(["review_year", "sentiment"]).size().reset_index(name="count")
    totals_by_year = trend_df.groupby("review_year").size().reset_index(name="total")
    sentiment_by_year = sentiment_by_year.merge(totals_by_year, on="review_year")
    sentiment_by_year["pct"] = (sentiment_by_year["count"] / sentiment_by_year["total"]) * 100

    return px.line(
        sentiment_by_year, x="review_year", y="pct", color="sentiment",
        markers=True, color_discrete_map=SENTIMENT_COLORS,
        labels={"pct": "% of Reviews", "review_year": "Year"},
    )


def topic_bar_chart(topics_df: pd.DataFrame, top_n: int = 10):
    topic_counts = topics_df[topics_df["topic"] != -1]["topic_name"].value_counts().head(top_n)
    fig = px.bar(x=topic_counts.values, y=topic_counts.index, orientation="h",
                 labels={"x": "Number of Reviews", "y": ""})
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return fig


def confusion_matrix_chart(matrix: list[list[int]]):
    labels = ["negative", "neutral", "positive"]
    return px.imshow(matrix, labels=dict(x="Predicted", y="Actual", color="Count"),
                      x=labels, y=labels, text_auto=True, color_continuous_scale="Blues")