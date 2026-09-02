from app.data.market import get_historical_data
from app.features.technical import calculate_technical_features
from app.database.connection import engine
from app.database.models import StockFeature

from sqlalchemy.dialects.postgresql import insert


def save_features(
    symbol: str,
    period: str = "2y",
):
    # 1. Get historical data
    df = get_historical_data(
        symbol=symbol,
        period=period,
    )

    # 2. Calculate technical indicators
    df = calculate_technical_features(df)

    # 3. Remove rows where indicators
    #    cannot yet be calculated
    df = df.dropna(
        subset=[
            "sma_20",
            "sma_50",
            "ema_20",
            "rsi_14",
            "macd",
            "macd_signal",
            "atr_14",
            "volatility_20",
            "volume_change",
            "next_day_return",
        ]
    )

    records = []

    for _, row in df.iterrows():

        records.append({
            "symbol": symbol,
            "timestamp": row["timestamp"].to_pydatetime(),

            "close": float(row["close"]),

            "sma_20": float(row["sma_20"]),
            "sma_50": float(row["sma_50"]),
            "ema_20": float(row["ema_20"]),

            "rsi_14": float(row["rsi_14"]),

            "macd": float(row["macd"]),
            "macd_signal": float(row["macd_signal"]),
            "macd_histogram": float(row["macd_histogram"]),

            "atr_14": float(row["atr_14"]),

            "volatility_20": float(row["volatility_20"]),

            "return_1d": float(row["return_1d"]),
            "return_5d": float(row["return_5d"]),

            "volume_change": float(row["volume_change"]),

            "next_day_return": float(row["next_day_return"]),
            "next_day_direction": int(row["next_day_direction"]),
        })

    if not records:
        print(f"No valid feature records for {symbol}")
        return

    stmt = insert(StockFeature).values(records)

    stmt = stmt.on_conflict_do_nothing(
        constraint="uq_feature_symbol_timestamp"
    )

    with engine.begin() as connection:
        connection.execute(stmt)

    print(
        f"Saved {len(records)} feature records for {symbol}"
    )