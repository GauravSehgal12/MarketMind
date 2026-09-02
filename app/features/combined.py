import pandas as pd

from app.data.market import get_historical_data
from app.features.technical import (
    calculate_technical_features,
)
from app.features.market_context import (
    get_market_context,
)


def create_feature_dataset(
    symbol: str,
    period: str = "2y",
) -> pd.DataFrame:

    # --------------------------------
    # Stock data
    # --------------------------------

    stock = get_historical_data(
        symbol=symbol,
        period=period,
    )

    stock = calculate_technical_features(
        stock
    )

    # --------------------------------
    # Market data
    # --------------------------------

    market = get_market_context(
        period=period
    )

    # --------------------------------
    # Merge stock + market
    # --------------------------------

    df = stock.merge(
        market,
        on="timestamp",
        how="inner",
    )

    # --------------------------------
    # Relative performance
    # --------------------------------

    df["nvda_vs_spy_1d"] = (
        df["return_1d"]
        - df["spy_return_1d"]
    )

    df["nvda_vs_qqq_1d"] = (
        df["return_1d"]
        - df["qqq_return_1d"]
    )

    # --------------------------------
    # Market momentum
    # --------------------------------

    df["market_breadth_score"] = (
        df["spy_return_5d"]
        + df["qqq_return_5d"]
    ) / 2

    # --------------------------------
    # Remove incomplete rows
    # --------------------------------

    df = df.dropna().reset_index(
        drop=True
    )

    return df