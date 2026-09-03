import pandas as pd
from sqlalchemy import text

from app.database.connection  import engine


def get_daily_news_features(
    symbol: str = "NVDA",
) -> pd.DataFrame:
    """
    Aggregate news sentiment by trading date.

    Returns one row per calendar date with:
    - average sentiment
    - maximum sentiment
    - minimum sentiment
    - article count
    - positive article ratio
    - negative article ratio
    """

    query = text("""
        SELECT
            DATE(published_at) AS timestamp,

            AVG(sentiment_score) AS news_sentiment_mean,

            MAX(sentiment_score) AS news_sentiment_max,

            MIN(sentiment_score) AS news_sentiment_min,

            COUNT(*) AS news_article_count,

            AVG(
                CASE
                    WHEN sentiment_label = 'positive'
                    THEN 1.0
                    ELSE 0.0
                END
            ) AS positive_news_ratio,

            AVG(
                CASE
                    WHEN sentiment_label = 'negative'
                    THEN 1.0
                    ELSE 0.0
                END
            ) AS negative_news_ratio

        FROM news_articles

        WHERE symbol = :symbol
          AND sentiment_score IS NOT NULL

        GROUP BY DATE(published_at)

        ORDER BY timestamp
    """)

    with engine.connect() as connection:
        result = connection.execute(
            query,
            {"symbol": symbol},
        )

        df = pd.DataFrame(
            result.mappings().all()
        )

    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    numeric_columns = [
        "news_sentiment_mean",
        "news_sentiment_max",
        "news_sentiment_min",
        "news_article_count",
        "positive_news_ratio",
        "negative_news_ratio",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column]
        )

    return df