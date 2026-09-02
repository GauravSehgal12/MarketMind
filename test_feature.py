from app.data.market import get_historical_data
from app.features.technical import calculate_technical_features


df = get_historical_data(
    symbol="NVDA",
    period="2y",
)

features = calculate_technical_features(df)

print(features[
    [
        "timestamp",
        "close",
        "sma_20",
        "sma_50",
        "ema_20",
        "rsi_14",
        "macd",
        "macd_signal",
        "atr_14",
        "volatility_20"
    ]
].tail(10))