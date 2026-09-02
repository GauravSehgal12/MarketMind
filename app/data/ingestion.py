from app.data.market import get_historical_data
from app.database.connection import engine
from app.database.models import StockPrice
from sqlalchemy.dialects.postgresql import insert


def save_stock_data(symbol: str, period: str = "2y"):

    df = get_historical_data(
        symbol=symbol,
        period=period,
    )

    records = []

    for _, row in df.iterrows():

        records.append({
            "symbol": symbol,
            "timestamp": row["timestamp"].to_pydatetime(),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "adj_close": float(row["adj_close"]),
            "volume": int(row["volume"]),
        })

    if not records:
        print(f"No data found for {symbol}")
        return

    stmt = insert(StockPrice).values(records)

    stmt = stmt.on_conflict_do_nothing(
        constraint="uq_stock_symbol_timestamp"
    )

    with engine.begin() as connection:
        connection.execute(stmt)

    print(
        f"Processed {len(records)} records for {symbol}"
    )