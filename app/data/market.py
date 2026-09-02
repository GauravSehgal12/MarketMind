import yfinance as yf
import requests
import pandas as pd

from app.config import settings


BASE_URL = "https://finnhub.io/api/v1"


def get_quote(symbol: str) -> dict:
    """Get current market quote from Finnhub."""

    url = f"{BASE_URL}/quote"

    params = {
        "symbol": symbol,
        "token": settings.finnhub_api_key,
    }

    response = requests.get(
        url,
        params=params,
        timeout=10,
    )

    response.raise_for_status()

    return response.json()





def get_historical_data(
    symbol: str,
    period: str = "2y",
) -> pd.DataFrame:

    ticker = yf.Ticker(symbol)

    df = ticker.history(
        period=period,
        auto_adjust=False,
    )

    if df.empty:
        raise ValueError(
            f"No historical data found for {symbol}"
        )

    # Handle yfinance MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    df = df[
        [
            "Date",
            "Adj Close",
            "Close",
            "High",
            "Low",
            "Open",
            "Volume",
        ]
    ]

    df = df.rename(
        columns={
            "Date": "timestamp",
            "Adj Close": "adj_close",
            "Close": "close",
            "High": "high",
            "Low": "low",
            "Open": "open",
            "Volume": "volume",
        }
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    ).dt.tz_localize(None)

    return df