import pandas as pd

from app.agents.news_features import get_daily_news_features


NEWS_COLUMNS = [
    "news_sentiment_mean",
    "news_sentiment_max",
    "news_sentiment_min",
    "news_article_count",
    "positive_news_ratio",
    "negative_news_ratio",
]


def add_news_features(
    price_features: pd.DataFrame,
    symbol: str,
) -> pd.DataFrame:
    """
    Add historical news sentiment features to the market
    feature dataset.

    News is shifted by one trading day to avoid target leakage.

    Example:

        News on Aug 31
              ↓
        Features available on Sep 1
              ↓
        Predict Sep 2 direction
    """

    # ----------------------------------
    # Validate input
    # ----------------------------------

    if price_features.empty:
        raise ValueError(
            "price_features is empty."
        )

    df = price_features.copy()

    # ----------------------------------
    # Ensure timestamp is datetime
    # ----------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # ----------------------------------
    # Get daily news features
    # ----------------------------------

    print(
        "\nLoading daily news features..."
    )
    print(
        "=============================="
    )

    news_df = get_daily_news_features(
        symbol=symbol
    )

    # ----------------------------------
    # Handle no news
    # ----------------------------------

    if news_df is None or news_df.empty:

        print(
            "No news data found."
        )

        for column in NEWS_COLUMNS:
            df[column] = 0.0

        df["news_available"] = 0

        return df

    # ----------------------------------
    # Ensure news timestamp is datetime
    # ----------------------------------

    news_df = news_df.copy()

    news_df["timestamp"] = pd.to_datetime(
        news_df["timestamp"]
    )

    news_df = news_df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # ----------------------------------
    # Remove duplicate dates
    # ----------------------------------

    news_df = (
        news_df
        .groupby("timestamp", as_index=False)
        .agg({
            "news_sentiment_mean": "mean",
            "news_sentiment_max": "max",
            "news_sentiment_min": "min",
            "news_article_count": "sum",
            "positive_news_ratio": "mean",
            "negative_news_ratio": "mean",
        })
    )

    # ----------------------------------
    # News availability
    # ----------------------------------

    news_df["news_available"] = (
        news_df["news_article_count"] > 0
    ).astype(int)

    # ----------------------------------
    # IMPORTANT:
    # Shift news by ONE day
    # ----------------------------------

    lag_columns = [
        "news_sentiment_mean",
        "news_sentiment_max",
        "news_sentiment_min",
        "news_article_count",
        "positive_news_ratio",
        "negative_news_ratio",
        "news_available",
    ]

    news_df[lag_columns] = (
        news_df[lag_columns].shift(1)
    )

    # ----------------------------------
    # Rename shifted features
    # ----------------------------------

    news_df = news_df.rename(
        columns={
            "news_sentiment_mean":
                "news_sentiment_mean_lag1",

            "news_sentiment_max":
                "news_sentiment_max_lag1",

            "news_sentiment_min":
                "news_sentiment_min_lag1",

            "news_article_count":
                "news_article_count_lag1",

            "positive_news_ratio":
                "positive_news_ratio_lag1",

            "negative_news_ratio":
                "negative_news_ratio_lag1",

            "news_available":
                "news_available_lag1",
        }
    )

    # ----------------------------------
    # Merge with market data
    # ----------------------------------

    df = df.merge(
        news_df,
        on="timestamp",
        how="left",
    )

    # ----------------------------------
    # Fill missing news values
    # ----------------------------------

    shifted_columns = [
        "news_sentiment_mean_lag1",
        "news_sentiment_max_lag1",
        "news_sentiment_min_lag1",
        "news_article_count_lag1",
        "positive_news_ratio_lag1",
        "negative_news_ratio_lag1",
    ]

    for column in shifted_columns:
        df[column] = df[column].fillna(0.0)

    df["news_available_lag1"] = (
        df["news_available_lag1"]
        .fillna(0)
        .astype(int)
    )

    # ----------------------------------
    # Display result
    # ----------------------------------

    print(
        "\nNews features added successfully."
    )

    print(
        "\nNews Feature Summary"
    )
    print(
        "===================="
    )

    print(
        df[
            [
                "timestamp",
                "news_sentiment_mean_lag1",
                "news_article_count_lag1",
                "positive_news_ratio_lag1",
                "negative_news_ratio_lag1",
                "news_available_lag1",
            ]
        ].tail(10).to_string(index=False)
    )

    return df