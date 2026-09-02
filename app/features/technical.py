import pandas as pd

from ta.momentum import RSIIndicator
from ta.trend import (
    EMAIndicator,
    MACD,
    SMAIndicator,
)
from ta.volatility import AverageTrueRange


def calculate_technical_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    # Make sure data is sorted chronologically
    df = df.sort_values("timestamp").reset_index(drop=True)

    # -----------------------------
    # Returns
    # -----------------------------

    df["return_1d"] = df["close"].pct_change()

    df["return_5d"] = df["close"].pct_change(5)

    # -----------------------------
    # Simple Moving Averages
    # -----------------------------

    df["sma_20"] = SMAIndicator(
        close=df["close"],
        window=20,
    ).sma_indicator()

    df["sma_50"] = SMAIndicator(
        close=df["close"],
        window=50,
    ).sma_indicator()

    # -----------------------------
    # Exponential Moving Average
    # -----------------------------

    df["ema_20"] = EMAIndicator(
        close=df["close"],
        window=20,
    ).ema_indicator()

    # -----------------------------
    # RSI
    # -----------------------------

    df["rsi_14"] = RSIIndicator(
        close=df["close"],
        window=14,
    ).rsi()

    # -----------------------------
    # MACD
    # -----------------------------

    macd = MACD(
        close=df["close"],
        window_fast=12,
        window_slow=26,
        window_sign=9,
    )

    df["macd"] = macd.macd()

    df["macd_signal"] = macd.macd_signal()

    df["macd_histogram"] = macd.macd_diff()

    # -----------------------------
    # ATR
    # -----------------------------

    atr = AverageTrueRange(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=14,
    )

    df["atr_14"] = atr.average_true_range()

    # -----------------------------
    # Volatility
    # -----------------------------

    df["volatility_20"] = (
        df["return_1d"]
        .rolling(window=20)
        .std()
    )

    # -----------------------------
    # Volume change
    # -----------------------------

    df["volume_change"] = (
        df["volume"].pct_change()
    )


    df["next_day_return"] = (
        df["close"].shift(-1) / df["close"] - 1
    )

    df["next_day_direction"] = (
        df["next_day_return"] > 0
    ).astype(int)

    

    return df
