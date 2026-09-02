import pandas as pd

from app.data.market import get_historical_data


def get_market_context(
    period: str = "2y",
) -> pd.DataFrame:

    # Download broader market data
    spy = get_historical_data(
        symbol="SPY",
        period=period,
    )

    qqq = get_historical_data(
        symbol="QQQ",
        period=period,
    )

    vix = get_historical_data(
        symbol="^VIX",
        period=period,
    )

    # --------------------------------
    # SPY features
    # --------------------------------

    spy_features = spy[
        ["timestamp", "close", "volume"]
    ].copy()

    spy_features["spy_return_1d"] = (
        spy_features["close"].pct_change()
    )

    spy_features["spy_return_5d"] = (
        spy_features["close"].pct_change(5)
    )

    spy_features["spy_volatility_20"] = (
        spy_features["spy_return_1d"]
        .rolling(20)
        .std()
    )

    spy_features = spy_features[
        [
            "timestamp",
            "spy_return_1d",
            "spy_return_5d",
            "spy_volatility_20",
        ]
    ]

    # --------------------------------
    # QQQ features
    # --------------------------------

    qqq_features = qqq[
        ["timestamp", "close"]
    ].copy()

    qqq_features["qqq_return_1d"] = (
        qqq_features["close"].pct_change()
    )

    qqq_features["qqq_return_5d"] = (
        qqq_features["close"].pct_change(5)
    )

    qqq_features = qqq_features[
        [
            "timestamp",
            "qqq_return_1d",
            "qqq_return_5d",
        ]
    ]

    # --------------------------------
    # VIX features
    # --------------------------------

    vix_features = vix[
        ["timestamp", "close"]
    ].copy()

    vix_features["vix_level"] = (
        vix_features["close"]
    )

    vix_features["vix_change"] = (
        vix_features["close"].pct_change()
    )

    vix_features = vix_features[
        [
            "timestamp",
            "vix_level",
            "vix_change",
        ]
    ]

    # --------------------------------
    # Merge
    # --------------------------------

    context = spy_features.merge(
        qqq_features,
        on="timestamp",
        how="inner",
    )

    context = context.merge(
        vix_features,
        on="timestamp",
        how="inner",
    )

    return context